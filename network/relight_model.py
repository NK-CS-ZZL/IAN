import os
import cv2
import torch
import importlib
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from network.loss import MSELoss
from collections import OrderedDict
from base_utils.utils import col_stitch
from network.arch import define_network
from network.base_model import BaseModel
from torch.nn.functional import interpolate
from base_utils.logger import get_root_logger
from dataset.data_utils import imwrite, tensor2img


loss_module = importlib.import_module('network.loss')
metric_module = importlib.import_module('metrics')

align_corners = False

class PyramidRelightingModel(BaseModel):
    """Pyramid Religting model for Religting."""

    def __init__(self, opt):
        super(PyramidRelightingModel, self).__init__(opt)

        self.num_layers = opt['layers']
        self.scale = opt['scale_factor']
        self.interp_mode = opt['interp_mode']
        self.loss_mask = opt['loss_mask'] if opt.get('loss_mask') != None else False
        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.losses = dict()
        self.best_cri = opt.get('val').get('best_metrics')
        self.best_cri2 = opt.get('val').get('best_metrics2')
        # self.with_depth = opt['with_depty']
        if opt['is_train']:
            self.decay_freq = opt.get('train').get('decay_freq')
            self.gamma = opt.get('train').get('gamma')
        
        
        self.multi_stage = opt.get('multi_stage')
        self.stage_weights = opt.get('stage_weights') 
        self.curr_stage = 0

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        for key in train_opt.keys():
            if key.endswith('_opt'):
                type = train_opt[key].pop('type')
                cri_cls = getattr(loss_module, type)
                for i in range(self.num_layers):
                    self.losses[type + str(i)] = cri_cls(**train_opt[key]).to(self.device)
        
        if len(self.losses) == 0:
            raise ValueError('Not Define Losses.')

        # set up optimizers and schedulers
        self.setup_optimizers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def lr_decay(self, curr_iter):
        if self.decay_freq and self.gamma and curr_iter != 0 and curr_iter % self.decay_freq == 0:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.gamma

    def __make_pyramid__(self, tensor, interp_mode=None):
        outputs = [tensor]
        resize_scale = 1. / self.scale
        for i in range(self.num_layers - 1):
            if interp_mode == None:
                sub_tensor = interpolate(tensor.detach(), scale_factor=resize_scale, mode=self.interp_mode)
            else:
                sub_tensor = interpolate(tensor.detach(), scale_factor=resize_scale, mode=interp_mode)
            resize_scale *= 1. / self.scale
            outputs.append(sub_tensor)
        return outputs

    def feed_data(self, data, is_training=True):
        self.input = data['input'].to(self.device)
        self.input = self.__make_pyramid__(self.input)
        
        if data.get('mask') != None:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask = torch.ones_like(data['input']).to(self.device)
        self.mask = self.__make_pyramid__(self.mask, 'nearest')

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt = self.__make_pyramid__(self.gt)

    def get_masked_output(self):
        return [(x+1)*y[:, :x.shape[1], :, :] - 1 for x, y in zip(self.output, self.mask)]

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.input)
        self.output = self.get_masked_output()

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        for loss_type in self.losses.keys():
            idx = int(loss_type[-1])
            loss = self.losses[loss_type](self.output[idx], self.gt[idx])
            if isinstance(loss, tuple):
                # perceptual loss
                l_total = l_total + loss[0]
                loss_dict[loss_type] = loss[0]
            else:
                l_total = l_total + loss
                loss_dict[loss_type] = loss

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.multi_stage != None and current_iter >= self.multi_stage[self.curr_stage] \
            and self.curr_stage < len(self.multi_stage)-1:
            self.curr_stage += 1
            print(f'change weight from {self.stage_weights[self.curr_stage-1]} to {self.stage_weights[self.curr_stage]}')
            for loss_type in self.losses.keys():
                # loss0->loss2, small->large
                idx = int(loss_type[-1])
                self.losses[loss_type].loss_weight = self.stage_weights[self.curr_stage][idx]

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.input)
            self.output = self.get_masked_output()
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['input_path'][0]))[0]
            self.feed_data(val_data, is_training=False)
            self.test()

            visuals = self.get_current_visuals()
            relight_img = tensor2img([visuals['result']], min_max=(-1, 1))
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))

            # tentative for out of GPU memory
            torch.cuda.empty_cache()
            
            if self.opt['val']['save_tb_img'] and img_name in self.opt['val']['save_tb_select']:
                input_img_rgb = tensor2img(self.input[0][:, :3, :, :], rgb2bgr=False, min_max=(-1, 1))
                relight_img_rgb = cv2.cvtColor(relight_img, cv2.COLOR_BGR2RGB)
                gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                img = col_stitch([input_img_rgb, relight_img_rgb, gt_img_rgb])
                tb_logger.add_images(f'eval_sample/{img_name}', np.array(img), current_iter, dataformats='HWC')

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt.get('val').get('suffix'):
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                    else:
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                imwrite(relight_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(relight_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                if self.best_cri != None and self.best_cri['name'] == metric:
                    self.save_best(metric, self.metric_results[metric], current_iter)
                elif self.best_cri2 != None and self.best_cri2['name'] == metric:
                    self.save_best2(metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        if self.best_cri != None:
            log_str += f'\t # best_{self.best_cri["name"]}: {self.best_cri["val"]:.4f}({self.best_cri["iter"]})\n'
        if self.best_cri2 != None:
            log_str += f'\t # best_{self.best_cri2["name"]}: {self.best_cri2["val"]:.4f}({self.best_cri2["iter"]})\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['input'] = self.input[0].detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[0].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)

    def save_best(self, name, val, current_iter):
        flag = False
        if self.best_cri['greater']:
            flag = True if self.best_cri['val'] < val else False
        else:
            flag = True if self.best_cri['val'] > val else False

        if flag:
            self.best_cri['val'] = val
            self.best_cri['iter'] = current_iter
            self.save(0, '{}_best'.format(name))

    def save_best2(self, name, val, current_iter):
        flag = False
        if self.best_cri2['greater']:
            flag = True if self.best_cri2['val'] < val else False
        else:
            flag = True if self.best_cri2['val'] > val else False

        if flag:
            self.best_cri2['val'] = val
            self.best_cri2['iter'] = current_iter
            self.save(0, '{}_best'.format(name))


class PyramidAnyRelightingModel(BaseModel):
    """Pyramid Religting model for Religting."""

    def __init__(self, opt):
        super(PyramidAnyRelightingModel, self).__init__(opt)

        self.num_layers = opt['layers']
        self.scale = opt['scale_factor']
        self.interp_mode = opt['interp_mode']
        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.losses = dict()
        self.best_cri = opt.get('val').get('best_metrics')
        self.best_cri2 = opt.get('val').get('best_metrics2')
        # self.with_depth = opt['with_depty']
        if opt['is_train']:
            self.decay_freq = opt.get('train').get('decay_freq')
            self.gamma = opt.get('train').get('gamma')
        
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        for key in train_opt.keys():
            if key.endswith('_opt'):
                type = train_opt[key].pop('type')
                cri_cls = getattr(loss_module, type)
                for i in range(self.num_layers):
                    self.losses[type + str(i)] = cri_cls(**train_opt[key]).to(self.device)
                    # if i == 2:
                    #     self.losses[type + str(i)].loss_weight = .5

        
        if len(self.losses) == 0:
            raise ValueError('Not Define Losses.')

        # set up optimizers and schedulers
        self.setup_optimizers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def lr_decay(self, curr_iter):
        if self.decay_freq and self.gamma and curr_iter != 0 and curr_iter % self.decay_freq == 0:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.gamma

    def __make_pyramid__(self, tensor, interp_mode=None):
        outputs = [tensor]
        resize_scale = 1. / self.scale
        for i in range(self.num_layers - 1):
            if interp_mode == None:
                sub_tensor = interpolate(tensor.detach(), scale_factor=resize_scale, mode=self.interp_mode)
            else:
                sub_tensor = interpolate(tensor.detach(), scale_factor=resize_scale, mode=interp_mode)
            resize_scale *= 1. / self.scale
            outputs.append(sub_tensor)
        return outputs

    def feed_data(self, data, is_training=True):
        self.input = data['input'].to(self.device)
        self.input_light = data['input_light'].to(self.device)
        self.input = self.__make_pyramid__(self.input)
        

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_light = data['gt_light'].to(self.device)
            self.gt = self.__make_pyramid__(self.gt)


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.input, self.gt_light)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        for loss_type in self.losses.keys():
            idx = int(loss_type[-1])
            loss = self.losses[loss_type](self.output[idx], self.gt[idx])
            if isinstance(loss, tuple):
                # perceptual loss
                l_total = l_total + loss[0]
                loss_dict[loss_type] = loss[0]
            else:
                l_total = l_total + loss
                loss_dict[loss_type] = loss

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.input, self.gt_light)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['input_path'][0]))[0]
            self.feed_data(val_data, is_training=False)
            self.test()

            visuals = self.get_current_visuals()
            relight_img = tensor2img([visuals['result']], min_max=(-1, 1))
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))

            # tentative for out of GPU memory
            torch.cuda.empty_cache()
            
            if self.opt['val']['save_tb_img'] and img_name in self.opt['val']['save_tb_select']:
                input_img_rgb = tensor2img(self.input[0][:, :3, :, :], rgb2bgr=False, min_max=(-1, 1))
                relight_img_rgb = cv2.cvtColor(relight_img, cv2.COLOR_BGR2RGB)
                gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                img = col_stitch([input_img_rgb, relight_img_rgb, gt_img_rgb])
                tb_logger.add_images(f'eval_sample/{img_name}', np.array(img), current_iter, dataformats='HWC')

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt.get('val').get('suffix'):
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                    else:
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                imwrite(relight_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(relight_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                if self.best_cri != None and self.best_cri['name'] == metric:
                    self.save_best(metric, self.metric_results[metric], current_iter)
                elif self.best_cri2 != None and self.best_cri2['name'] == metric:
                    self.save_best2(metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        if self.best_cri != None:
            log_str += f'\t # best_{self.best_cri["name"]}: {self.best_cri["val"]:.4f}({self.best_cri["iter"]})\n'
        if self.best_cri2 != None:
            log_str += f'\t # best_{self.best_cri2["name"]}: {self.best_cri2["val"]:.4f}({self.best_cri2["iter"]})\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['input'] = self.input[0].detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[0].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)

    def save_best(self, name, val, current_iter):
        flag = False
        if self.best_cri['greater']:
            flag = True if self.best_cri['val'] < val else False
        else:
            flag = True if self.best_cri['val'] > val else False

        if flag:
            self.best_cri['val'] = val
            self.best_cri['iter'] = current_iter
            self.save(0, '{}_best'.format(name))

    def save_best2(self, name, val, current_iter):
        flag = False
        if self.best_cri2['greater']:
            flag = True if self.best_cri2['val'] < val else False
        else:
            flag = True if self.best_cri2['val'] > val else False

        if flag:
            self.best_cri2['val'] = val
            self.best_cri2['iter'] = current_iter
            self.save(0, '{}_best'.format(name))

class DeepPortraitRelightingModel(BaseModel):
    """Pyramid Religting model for Religting."""

    def __init__(self, opt):
        super(DeepPortraitRelightingModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.losses = dict()
        self.light_loss = MSELoss(loss_weight=1)
        self.feat_loss = MSELoss(loss_weight=0.5)
        self.best_cri = opt.get('val').get('best_metrics')
        self.best_cri2 = opt.get('val').get('best_metrics2')
        if opt['is_train']:
            self.decay_freq = opt.get('train').get('decay_freq')
            self.gamma = opt.get('train').get('gamma')
        
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        for key in train_opt.keys():
            if key.endswith('_opt'):
                type = train_opt[key].pop('type')
                cri_cls = getattr(loss_module, type)
                self.losses[type] = cri_cls(**train_opt[key]).to(self.device)
                   
        
        if len(self.losses) == 0:
            raise ValueError('Not Define Losses.')

        # set up optimizers and schedulers
        self.setup_optimizers()



    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def lr_decay(self, curr_iter):
        if self.decay_freq and self.gamma and curr_iter != 0 and curr_iter % self.decay_freq == 0:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.gamma


    def feed_data(self, data, is_training=True):
        self.input = data['input'].to(self.device)
        self.input_light = data['input_light'].to(self.device)
        b, c = self.input_light.shape
        self.input_light = self.input_light.view(b, c, 1, 1)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_light = data['gt_light'].to(self.device)
            self.gt_light = self.gt_light.view(b, c, 1, 1)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, self.out_feat, self.out_light, self.ori_feat \
                        = self.net_g(self.input, self.gt_light, 4, self.gt)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        for loss_type in self.losses.keys():
            loss = self.losses[loss_type](self.output, self.gt)
            if isinstance(loss, tuple):
                # perceptual loss
                l_total = l_total + loss[0]
                loss_dict[loss_type] = loss[0]
            else:
                l_total = l_total + loss
                loss_dict[loss_type] = loss

        loss_dict['feat_loss'] = self.feat_loss(self.out_feat, self.ori_feat)
        loss_dict['light_loss'] = self.light_loss(self.out_light, self.gt_light)
        l_total += loss_dict['feat_loss'] + loss_dict['light_loss']

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output, self.out_feat, self.out_light, self.ori_feat \
                                 = self.net_g(self.input, self.gt_light, 4, None)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['input_path'][0]))[0]
   
            self.feed_data(val_data, is_training=False)
            self.test()

            visuals = self.get_current_visuals()
            relight_img = tensor2img([visuals['result']], min_max=(-1, 1))
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))

            # tentative for out of GPU memory
            torch.cuda.empty_cache()
            
            if self.opt['val']['save_tb_img'] and img_name in self.opt['val']['save_tb_select']:
                input_img_rgb = tensor2img(self.input[0][:, :3, :, :], rgb2bgr=False, min_max=(-1, 1))
                relight_img_rgb = cv2.cvtColor(relight_img, cv2.COLOR_BGR2RGB)
                gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                img = col_stitch([input_img_rgb, relight_img_rgb, gt_img_rgb])
                tb_logger.add_images(f'eval_sample/{img_name}', np.array(img), current_iter, dataformats='HWC')

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt.get('val').get('suffix'):
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                    else:
                        save_img_path = os.path.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                imwrite(relight_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(relight_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                if self.best_cri != None and self.best_cri['name'] == metric:
                    self.save_best(metric, self.metric_results[metric], current_iter)
                elif self.best_cri2 != None and self.best_cri2['name'] == metric:
                    self.save_best2(metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        if self.best_cri != None:
            log_str += f'\t # best_{self.best_cri["name"]}: {self.best_cri["val"]:.4f}({self.best_cri["iter"]})\n'
        if self.best_cri2 != None:
            log_str += f'\t # best_{self.best_cri2["name"]}: {self.best_cri2["val"]:.4f}({self.best_cri2["iter"]})\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['input'] = self.input[0].detach().cpu()
        out_dict['result'] = self.output[0].detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[0].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)

    def save_best(self, name, val, current_iter):
        flag = False
        if self.best_cri['greater']:
            flag = True if self.best_cri['val'] < val else False
        else:
            flag = True if self.best_cri['val'] > val else False

        if flag:
            self.best_cri['val'] = val
            self.best_cri['iter'] = current_iter
            self.save(0, '{}_best'.format(name))

    def save_best2(self, name, val, current_iter):
        flag = False
        if self.best_cri2['greater']:
            flag = True if self.best_cri2['val'] < val else False
        else:
            flag = True if self.best_cri2['val'] > val else False

        if flag:
            self.best_cri2['val'] = val
            self.best_cri2['iter'] = current_iter
            self.save(0, '{}_best'.format(name))









import torch
import numpy as np
from torch.utils import data as data 
from base_utils.utils import load_depth
from dataset.transforms import augment, multi_random_crop
from dataset.data_utils import paired_paths_from_folder, img2tensor, imread


class PairedImageWithAuxiliaryDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageWithAuxiliaryDataset, self).__init__()
        self.opt = opt
        self.mask = False
        self.return_mask = False
        self.paths = []
        if opt.get('mask') != None:
            self.mask = opt['mask']['enable']
            self.return_mask = opt['mask']['return']
        if opt.get('test_mode') == None:
            self.test_mode = False
        else:
            self.test_mode = opt['test_mode']

        if isinstance(self.opt['auxiliary_dim'], list) == False:
            self.opt['auxiliary_dim'] = [self.opt['auxiliary_dim']]
        if isinstance(opt['dataroot_auxiliary'], list) == False:
            opt['dataroot_auxiliary'] = [[opt['dataroot_auxiliary']]]
        elif isinstance(opt['dataroot_auxiliary'][0], list) == False:
            opt['dataroot_auxiliary'] = [opt['dataroot_auxiliary']]


        for i in range(len(opt['dataroot_input'])):
            if self.test_mode:
                self.input_folder = opt['dataroot_input'][i]
                self.aux_folders = opt['dataroot_auxiliary'][i]
            else:
                self.gt_folder, self.input_folder = opt['dataroot_gt'][i], opt['dataroot_input'][i]
                self.aux_folders = opt['dataroot_auxiliary'][i]
        
            if 'filename_tmpl' in opt:
                self.filename_tmpl = opt['filename_tmpl']
            else:
                self.filename_tmpl = '{}'

            if self.test_mode:
                self.paths += paired_paths_from_folder(
                    [self.input_folder, None, self.aux_folders], ['input', None, 'aux'],
                    self.filename_tmpl)
            else:
                self.paths += paired_paths_from_folder(
                    [self.input_folder, self.gt_folder, self.aux_folders], ['input', 'gt', 'aux'],
                    self.filename_tmpl)
       

    def __getitem__(self, index):
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [-1, 1], float32.
        if not self.test_mode:
            gt_path = self.paths[index]['gt_path']
            img_gt = imread(gt_path)
        input_path = self.paths[index]['input_path']
        img_input = imread(input_path)
        sum_aux_input = []
        for dim, aux_paths in zip(self.opt['auxiliary_dim'], self.paths[index]['aux_paths']):
            aux_path = aux_paths
            if dim == 1:
                aux_input = load_depth(aux_path)
            else:
                aux_input = np.load(aux_path)
            if len(aux_input.shape) == 2:
                aux_input = aux_input[:, :, np.newaxis]
            sum_aux_input.append(aux_input)
            
        sum_aux_input = np.concatenate(sum_aux_input, axis=2).astype(np.float32)
        # augmentation for training
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_input, sum_aux_input = augment([img_gt, img_input, sum_aux_input], self.opt['use_flip'], self.opt['use_rot'], self.opt.get('use_color'))

        # crop imgs
            if self.opt['gt_size'] > 0:
                (img_gt, img_input, sum_aux_input), mask =  multi_random_crop([img_gt, img_input, sum_aux_input], self.opt['gt_size'])        

        # BGR to RGB, HWC to CHW, numpy to tensor
        if self.test_mode:
            img_input, sum_aux_input = img2tensor([img_input, sum_aux_input],
                                    bgr2rgb=True,
                                    float32=True)
            img_gt = None
            gt_path = None
        else:
            img_gt, img_input, sum_aux_input = img2tensor([img_gt, img_input, sum_aux_input],
                                    bgr2rgb=True,
                                    float32=True)
        img_input = torch.cat([img_input, sum_aux_input], axis=0)
        if self.mask == True and self.return_mask == False:
            img_input = img_input[:-1, ...]
        if img_gt != None:
            return {
                'input': img_input,
                'gt': img_gt,
                'input_path': input_path,
                'gt_path': gt_path
            }
        else:
            return {
                'input': img_input,
                'input_path': input_path,
            }

    def __len__(self):
        return len(self.paths)


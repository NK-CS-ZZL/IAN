import os
import torch
import logging
from train import parse_options
from network import create_model
from options.yaml_opt import dict2str
from dataset import create_dataloader, create_dataset
from base_utils.utils import get_time_str, make_exp_dirs
from base_utils.logger import get_root_logger, get_env_info



def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_id']
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = os.path.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='relighting', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)
    #model = nn.DataParallel(model).cuda()
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'])


if __name__ == '__main__':
    main()
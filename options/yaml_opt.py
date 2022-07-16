import yaml
from collections import OrderedDict
import os


def dict2str(opt, indent_level=1):
    """dict to string for printing options.
    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.
    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def ordered_yaml():
    """Support OrderedDict for yaml.
    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.
    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.
    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train
    if opt.get("mask_loss") == True:
        opt['dataset']['train']['mask']['enable'] = True
        opt['dataset']['train']['mask']['return'] = True

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for several datasets, e.g., test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            if isinstance(dataset['dataroot_gt'], list) == False:
                dataset['dataroot_gt'] = [dataset['dataroot_gt']]
            dataset['dataroot_gt'] = [os.path.expanduser(x) for x in dataset['dataroot_gt']]
        if dataset.get('dataroot_input') is not None:
            if isinstance(dataset['dataroot_input'], list) == False:
                dataset['dataroot_input'] = [dataset['dataroot_input']]
            dataset['dataroot_input'] = [os.path.expanduser(x) for x in dataset['dataroot_input']]
        if dataset.get('dataroot_ref') is not None:
            if isinstance(dataset['dataroot_ref'], list) == False:
                dataset['dataroot_ref'] = [dataset['dataroot_ref']]
            dataset['dataroot_ref'] = [os.path.expanduser(x) for x in dataset['dataroot_ref']]

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = os.path.expanduser(val)
    opt['path']['root'] = os.path.abspath(
        os.path.join(__file__, os.path.pardir, os.path.pardir))
    if is_train:
        experiments_root = os.path.join('.', 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_states'] = os.path.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = os.path.join(experiments_root,
                                                'visualization')
        if opt['logger']['use_tb_logger'] == True:
            opt['path']['tb_logger'] = os.path.join(experiments_root, 'tb_logger')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = os.path.join(results_root, 'visualization')

    return opt

if __name__ == '__main__':
    parse('./opt.yml')
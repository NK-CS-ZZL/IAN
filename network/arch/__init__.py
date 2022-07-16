import importlib
import os


# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = os.path.dirname(os.path.abspath(__file__))
arch_filenames = [
    os.path.splitext(os.path.basename(v))[0] for v in os.listdir(arch_folder)
    if v.endswith('_arch.py')
]

# import all the arch modules
_arch_modules = [
    importlib.import_module(f'network.arch.{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.
    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.
    Returns:
        class: Instantiated class.
    """
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    network_type = opt.pop('type')
    print(network_type)
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net

if __name__ == '__main__':
    opt = {'type': 'FCN'}
    net = define_network(opt)
    print(net)
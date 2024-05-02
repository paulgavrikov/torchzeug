from functools import reduce
import torch

# from https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/6
def get_module_by_name(module, access_string):
    """
    Get a module by its name.
    
    Args:
        module: The module to search in.
        access_string: The name of the module to get.
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def set_module_by_name(module, access_string, value):
    """
    Set a module by its name.
    
    Args:
        module: The module to search in.
        access_string: The name of the module to set.
        value: The value to set the module to.
    """
    names = access_string.split(sep='.')
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def get_module_parameters(module, flatten=False):
    """
    Get the parameters of a module.
    
    Args:
        module: The module to get the parameters of.
        flatten: Whether to flatten the parameters into tensor vector.

    Returns:
        The parameters of the module.
    """
    p = list(module.parameters())
    if flatten:
        p = torch.cat([x.view(-1) for x in p])
    return p


def freeze_module(module):
    """
    Freeze a module.
    
    Args:
        module: The module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """
    Unfreeze a module.
    
    Args:
        module: The module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True
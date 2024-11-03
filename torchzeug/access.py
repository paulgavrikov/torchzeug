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


class LayerHook():
    
    def __init__(self):
        self.storage = None
        self.hook_handle = None

    def pull(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        data = self.storage
        self.storage = None
        return data

    def register_hook(self, module, store_input=True):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.storage = None
        def hook(_, inp, out):
            if store_input:
                self.storage = inp
            else:
                self.storage = out
        self.hook_handle = module.register_forward_hook(hook)


def get_activations(model, x, layer, get_input=False):
    """
    Get the activations of a layer.

    Args:
        model: The model to get the activations from.
        x: The input to the model.
        layer: The module to get the activations from.
        get_input: Whether to get the inputs (True) or outputs (False) to the layer. Default: False.
    """
    hook = LayerHook()
    hook.register_hook(module=layer, store_input=get_input)
    model(x)
    return hook.pull()


def estimate_model_device(model):
    """
    Estimate the device of a given model.

    Args:
        model: The model to estimate the device of.

    Returns:
        The device (str) of the model.
    """
    from collections import Counter

    device_counts = Counter([str(p.device) for p in model.parameters()])
    device = device_counts.most_common(1)[0][0]
    return device
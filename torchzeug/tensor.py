import torch


def tensor_map(func, iterable):
    """
    Apply a function to each element of an iterable and return a tensor of the results.

    Args:
        func: The function to apply to each element.
        iterable: The iterable to map over.
    """
    return torch.stack([func(x) for x in iterable])
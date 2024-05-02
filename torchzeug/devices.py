import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import numpy as np


def get_used_gpu_memory():
    """
    Get the amount of memory used on each GPU.

    Returns:
        A list of the amount of memory used on each GPU.
    """    
    nvmlInit()
    stats = []
    for i in range(torch.cuda.device_count()):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        stats.append(info.used)
    return stats


def autoselect_device():
    """
    Automatically select the best device to use.

    Strategy: If a GPU is available, use the GPU with the least memory used. If no GPU is available, use MPS if available. Otherwise, use the CPU.

    Returns:
        The device to use.
    """
    best_is_gpu = False

    device = "cpu"

    try:
        mps_available = torch.backends.mps.is_available()
    except:
        mps_available = False

    if torch.cuda.is_available():
        best_is_gpu = True
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"

    if best_is_gpu:
        best_device = f"cuda:{np.argmin(get_used_gpu_memory())}"
        device = best_device

    return device
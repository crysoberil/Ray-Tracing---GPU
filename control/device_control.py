import torch


def _get_gpu_device():
    return None

_device = _get_gpu_device()


def to_gpu_if_possible(tensor):
    if _device is not None:
        return tensor.to(_device)
    return tensor


def extract_value_from_tensor(tensor):
    return tensor.data.cpu().numpy()


def get_device_float32_array(shape, fill):
    if _device is not None:
        return torch.cuda.FloatTensor(*shape).fill_(fill)
    else:
        return torch.FloatTensor(*shape).fill_(fill)


def get_device_int32_array(shape, fill):
    if _device is not None:
        return torch.cuda.IntTensor(*shape).fill_(fill)
    else:
        return torch.IntTensor(*shape).fill_(fill)


def get_device_int64_array(shape, fill):
    if _device is not None:
        return torch.cuda.LongTensor(*shape).fill_(fill)
    else:
        return torch.LongTensor(*shape).fill_(fill)


def get_device_uint8_array(shape, fill):
    if _device is not None:
        return torch.cuda.ByteTensor(*shape).fill_(fill)
    else:
        return torch.ByteTensor(*shape).fill_(fill)
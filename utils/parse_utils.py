from control import device_control
import torch


def parse_tensor_from_string_toks(toks):
    toks = [float(elm) for elm in toks]
    toks = torch.FloatTensor(toks)
    return device_control.to_gpu_if_possible(toks)
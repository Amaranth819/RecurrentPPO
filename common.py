import torch
import numpy as np

'''
    Cast a np array to tensor
'''
def np_to_tensor(np_array, device = torch.device('cpu')):
    return torch.from_numpy(np_array).float().to(device)


'''
    Cast a tensor to np array
'''
def tensor_to_np(tensor : torch.Tensor):
    return tensor.detach().cpu().numpy()
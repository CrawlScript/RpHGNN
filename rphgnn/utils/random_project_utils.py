from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from rphgnn.global_configuration import global_config





def torch_normalize_l2(x):
    return F.normalize(x, dim=-1)

torch_normalize = torch_normalize_l2

def get_reversed_etype(etype):
    
    if etype[0] == etype[2]:
        return etype

    reversed_etype_ = etype[1][2:] if etype[1].startswith("r.") else "r.{}".format(etype[1])
    return (etype[2], reversed_etype_, etype[0])


def create_func_torch_random_project_create_kernel_sparse(s=3.0):

    def torch_random_project_create_kernel_sparse(x, units, input_units=None, generator=None):

        if input_units is None:
            input_units = x.size(-1)
        shape = [input_units, units]

        stddev = 1.0

        if generator is None:
            probs = torch.rand(shape)
        else:
            print("generate fast random projection kernel with generator")
            probs = torch.rand(shape, generator=generator)


        fill = torch.ones(shape) * torch.sqrt(torch.tensor(s))  * stddev

        kernel = torch.zeros(shape)
        kernel = torch.where(probs >= (1.0 - 0.5 / s), fill, kernel)
        kernel = torch.where(probs < (0.5 / s), -fill, kernel)

        return kernel
    
    return torch_random_project_create_kernel_sparse


def torch_random_project_create_kernel_xavier(x, units, input_units=None, generator=None):
    if input_units is None:
            input_units = x.size(-1)
    shape = [input_units, units]
    stddev = torch.sqrt(torch.tensor(2.0 / (shape[0] + shape[1])))
    kernel = torch.randn(shape) * stddev
    return kernel


def torch_random_project_create_kernel_xavier_no_norm(x, units, input_units=None, generator=None):
    if input_units is None:
            input_units = x.size(-1)
    shape = [input_units, units]

    stddev = 1.0
    kernel = torch.randn(shape) * stddev
    return kernel


def torch_random_project_common(x, units, activation=False, norm=True, kernel=None, generator=None):

    if kernel is None:
        kernel = global_config.torch_random_project_create_kernel(x, units, generator=generator)

    h = x @ kernel

    if norm:
        h = torch_normalize(h)

    return h


global_config.torch_random_project = torch_random_project_common
global_config.torch_random_project_create_kernel = create_func_torch_random_project_create_kernel_sparse(s=3.0)




def torch_random_project_then_sum(x_list, units, norm=True, generator=None):
    h_list = [global_config.torch_random_project(x, units, norm=norm, generator=generator) 
        for x in x_list]
    h = torch.stack(h_list, dim=0).sum(dim=0)
    return h



def torch_random_project_then_mean(x_list, units, norm=True, num_samplings=None):
    
    h_list = [global_config.torch_random_project(x, units, norm=norm) 
        for x in x_list]
    h = torch.stack(h_list, dim=0).mean(dim=0)

    return h



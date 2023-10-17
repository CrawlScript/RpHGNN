# coding=utf-8
import torch

class GlobalConfig(object):
    def __init__(self) -> None:

        self.embedding_generator = None
        self.rand_proj_generator = None

        self.torch_random_project = None
        self.torch_random_project_create_kernel = None



global_config = GlobalConfig()
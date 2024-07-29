# coding=utf-8

import torch
import torch.nn as nn
from itertools import chain
import torch.nn.functional as F

from rphgnn.layers.torch_train_model import CommonTorchTrainModel



class MyPReLU(nn.Module):

    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()

        # use alpha instead of weight
        self.alpha = nn.parameter.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.alpha)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()

        self.func = func

    def forward(self, x):
        return self.func(x)
        
def create_act(name=None):
    if name == "softmax":
        return nn.Softmax(dim=-1)
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return Lambda(lambda x: x)
    else:
        raise Exception()


class MyLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class MyConv1d(nn.Conv1d):
    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class MyMLPConv1d(nn.Module):
    """
    another implementation of MLPConv1d
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = MyLinear(in_channels, out_channels)

    def forward(self, x):
        h = torch.permute(x, (0, 2, 1))
        h = self.linear(h)
        h = torch.permute(h, (0, 2, 1))
        h = h.contiguous()
        return h


class MLP(nn.Module):

    def __init__(self,
                 units_list,
                 drop_rate,
                 input_shape,
                 kernel_regularizer=None,
                 activation=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_fc = None
        self.decoder_fc = None

        self.drop_rate = drop_rate
        self.kernel_regularizer = kernel_regularizer
        self.relation_encoders = None

        self.activation = activation


        input_units = input_shape[-1]

        units_list = [input_units] + units_list
        
        self.encoder_fc = nn.Sequential(
                *(list(chain(*

                            [[MyLinear(units_list[i], units_list[i + 1]),
                                 MyPReLU(),
                                 nn.Dropout(drop_rate)]
                                 for i, _ in enumerate(units_list[:-2])]
                            ))
                +
                ([
                    MyLinear(units_list[-2], units_list[-1]),
                    MyPReLU(),
                    nn.Dropout(drop_rate)
                ] if self.activation is None else 
                [
                    MyLinear(units_list[-2], units_list[-1]),
                    create_act(activation)
                ]) 
                
                )

        )


    def forward(self, x):
        return self.encoder_fc(x)



class GroupEncoders(nn.Module):

    def __init__(self,
                 filters_list,
                 drop_rate,
                 input_shape,
                 kernel_regularizer=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hop_encoders = None
        self.filters_list = filters_list
        self.drop_rate = drop_rate
        self.kernel_regularizer = kernel_regularizer
        self.real_filters_list = None

        num_groups = len(input_shape)
        self.group_sizes = [group_shape[1] for group_shape in input_shape]
        self.real_filters_list = [self._get_real_filters(i) for i in range(num_groups)]

        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                MyConv1d(group_size, real_filters, 1, stride=1),
                # # if too slow, comment MyConv1d (above) and uncomment MyMLPConv1d (below)
                # MyMLPConv1d(group_size, real_filters),
                Lambda(lambda x: x.view(x.size(0), -1))
            )
            for _, (group_size, real_filters) in enumerate(zip(self.group_sizes, self.real_filters_list))
        ])


    def _get_real_filters(self, i):
        
        if self.group_sizes[i] == 1:
            return 1
        elif isinstance(self.filters_list, list):
            return self.filters_list[i]
        else:
            return self.filters_list
 

    def forward(self, x_group_list):
        group_h_list = []

        for i, (x_group, group_encoder) in enumerate(zip(x_group_list, self.group_encoders)):

            h = x_group
            group_h = group_encoder(h)
            group_h_list.append(group_h)

        return group_h_list



class MultiGroupFusion(nn.Module):

    def __init__(self,
                 group_units_list,
                 global_units_list,
                 merge_mode,
                 drop_rate,
                 activation=None,
                 input_shape=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.group_fc_list = None
        self.global_fc = None

        self.group_units_list = group_units_list
        self.global_units_list = global_units_list
        self.merge_mode = merge_mode
        self.drop_rate = drop_rate

        self.activation = activation

        self.use_shared_group_fc = False
        self.group_encoder_mode = "common" 

        num_groups = len(input_shape)
        self.num_groups = num_groups

        self.group_fc_list = nn.ModuleList([
            MLP(group_units_list, drop_rate, input_shape=group_input_shape)
            for group_input_shape in input_shape
        ])

        if merge_mode in ["mean", "free"]:
            global_input_shape = [-1, group_units_list[-1]]
        elif merge_mode == "concat":
            global_input_shape = [-1, group_units_list[-1] * num_groups]
        else:
            raise Exception("wrong merge mode: ", merge_mode)

        self.global_fc = MLP(self.global_units_list, drop_rate=self.drop_rate, activation=self.activation, input_shape=global_input_shape)


    def forward(self, inputs):

        x_list = inputs
        group_h_list = [group_fc(x) for x, group_fc in zip(x_list, self.group_fc_list)]

        if self.merge_mode == "mean":
            global_h = torch.stack(group_h_list, dim=0).mean(dim=0)
        elif self.merge_mode == "concat":
            global_h = torch.concat(group_h_list, dim=-1)
        else:
            raise Exception("wrong merge mode: ", self.merge_mode)

        h = self.global_fc(global_h)

        return h



class RpHGNNEncoder(CommonTorchTrainModel):

    def __init__(self,
                 filters_list,
                 group_units_list,
                 global_units_list,
                 merge_mode,
                 input_shape,
                 *args,
                 input_drop_rate=0.0,
                 drop_rate=0.0,
                 activation=None,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.input_dropout = nn.Dropout(input_drop_rate)
        self.input_drop_rate = input_drop_rate

        group_encoders_input_shape = input_shape
        self.group_encoders = GroupEncoders(filters_list, drop_rate, group_encoders_input_shape)

        multi_group_fusion_input_shape = [[-1, group_input_shape[-1] * filters] 
                                          for group_input_shape, filters in zip(group_encoders_input_shape, self.group_encoders.real_filters_list)]
        self.multi_group_fusion = MultiGroupFusion(group_units_list, global_units_list, merge_mode, drop_rate, activation=activation, input_shape=multi_group_fusion_input_shape)

    def forward(self, inputs):

        x_group_list = inputs
        dropped_x_group_list = [F.dropout(x_group, self.input_drop_rate, training=self.training, inplace=False) for x_group in x_group_list]

        h_list = self.group_encoders(dropped_x_group_list)
        h = self.multi_group_fusion(h_list)

        return h

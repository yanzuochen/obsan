import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy

# from tqdm import tqdm

# from models import encoder_64

valid_instance_list = [
    nn.Linear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.RNN, nn.LSTM, nn.GRU
]

def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            # or isinstance(module, nn.Conv3d)
            # or isinstance(module, nn.RNN)
            # or isinstance(module, nn.LSTM)
            # or isinstance(module, nn.GRU)
            )

def iterate_module(name, module, name_list, module_list):
    if is_valid(module):
        return name_list + [name], module_list + [module]
    else:

        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list

def get_model_layers(model):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [])
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])

    return layer_dict


def get_layer_output_sizes(model, data, layer_name=None):
    output_sizes = {}
    hooks = []

    layer_dict = get_model_layers(model)

    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])
        # output_sizes[m_key] = list(output.size())

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        if type(data) is tuple:
            model(*data)
        else:
            model(data)
    finally:
        for h in hooks:
            h.remove()
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])

    return output_sizes


def get_layer_output(model, data, threshold=0.5, force_relu=False, layer_name=None):
    with torch.no_grad():
        layer_dict = get_model_layers(model)

        layer_output_dict = {}
        def hook(module, input, output):
            module_idx = len(layer_output_dict)
            m_key = list(layer_dict)[module_idx]
            if force_relu:
                output = F.relu(output)
            layer_output_dict[m_key] = output # (N, K, H, W) or (N, K)

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        # layer_output_list = []
        for layer, output in layer_output_dict.items():
            assert len(output.size()) == 2 or len(output.size()) == 4
            if len(output.size()) == 4: # (N, K, H, w)
                output = output.mean((2, 3))
            # scaled_output = scale(output)
            layer_output_dict[layer] = output.detach()
            # print(layer, ': ', output.size())
            # mask_index = scaled_output > threshold
            # print(mask_index)
            # masked_output = mask_index * scaled_output
            # layer_output_dict[layer] = masked_output
            # layer_output_list.append(mask_index)
        # return final_out, layer_output_list
        return layer_output_dict



if __name__ == '__main__':
    pass

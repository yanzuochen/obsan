import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset

from tqdm import tqdm
# import progressbar
from pyflann import FLANN

import neuron_coverage as tool
import classifier as cifar10_models
from data_loader import *


# def scale(out, rmax=1, rmin=0):
#     output_std = (out - out.min()) / (out.max() - out.min())
#     output_scaled = output_std * (rmax - rmin) + rmin
#     return output_scaled

def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    # out_max = out.max()
    # out_min = out.min()
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


class LayerNCS(object):
    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)

class LayerNC(object):
    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            mask_index = layer_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)

class LayerKMN(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_multisec_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_multisec_dict[layer_name] = torch.zeros((num_neuron, k + 1)).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'multisec_cove_dict': self.coverage_multisec_dict
        }
        self.current = 0

    def build(self, data_list):
        print('Building Range...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.set_range(data)
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        multisec_cove_dict = {}
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(num_neuron, self.k + 1).cuda().type(torch.cuda.BoolTensor)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index
            # (1, k), index 0 indicates out-of-range output

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cove_dict[layer_name] = multisec_covered | self.coverage_multisec_dict[layer_name]

        return {
            'multisec_cove_dict': multisec_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        multisec_covered = all_covered_dict['multisec_covered']

        num_neuron = multisec_covered.size(0)
        multisec_num_covered = torch.sum(multisec_covered[:, 1:])
        multisec_rate = multisec_num_covered / (num_neuron * self.k)

        return multisec_rate.item()

    def all_coverage(self, all_cove_dict):
        multisec_cove_dict = all_cove_dict['multisec_cove_dict']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total
        return multisec_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_multisec_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)


class LayerSNA(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_upper_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_upper_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'upper_cove_dict': self.coverage_upper_dict
        }
        self.current = 0

    def build(self, data_list):
        print('Building Range...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.set_range(data)
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cove_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'upper_cove_dict': upper_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        upper_covered = all_covered_dict['upper_covered']
        upper_rate = upper_covered.sum() / len(upper_covered)
        return upper_rate.item()

    def all_coverage(self, all_cove_dict):
        upper_cove_dict = all_cove_dict['upper_cove_dict']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     torch.save(self.coverage_upper_dict, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)

class LayerNBC(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_lower_dict = {}
        self.coverage_upper_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_lower_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.coverage_upper_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.range_dict[layer_name] = [torch.ones(num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
        self.coverage_dict = {
            'lower_cove_dict': self.coverage_lower_dict,
            'upper_cove_dict': self.coverage_upper_dict
        }
        self.current = 0

    def build(self, data_list):
        print('Building Range...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.set_range(data)
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cove_dict = self.calculate(data)
        self.update(all_cove_dict)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]

            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_lower_dict[layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'lower_cove_dict': lower_cove_dict,
            'upper_cove_dict': upper_cove_dict
        }

    def update(self, all_cove_dict, delta=None):
        for k in all_cove_dict.keys():
            self.coverage_dict[k] = all_cove_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cove_dict)

    def coverage(self, all_covered_dict):
        lower_covered = all_covered_dict['lower_covered']
        upper_covered = all_covered_dict['upper_covered']

        lower_rate = lower_covered.sum() / len(lower_covered)
        upper_rate = upper_covered.sum() / len(upper_covered)

        return (lower_rate + upper_rate).item() / 2

    def all_coverage(self, all_cove_dict):
        lower_cove_dict = all_cove_dict['lower_cove_dict']
        upper_cove_dict = all_cove_dict['upper_cove_dict']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cove_dict.keys():
            lower_covered = lower_cove_dict[layer_name]
            upper_covered = upper_cove_dict[layer_name]

            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()

            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = lower_cove / lower_total
        upper_rate = upper_cove / upper_total
        return (lower_rate + upper_rate).item() / 2

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    # def save(self, path):
    #     state = {
    #         'lower': self.coverage_lower_dict,
    #         'upper': self.coverage_upper_dict
    #     }
    #     torch.save(state, path)

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']

        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)

class LayerTopK(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
        self.current = 0

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).cuda()
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1

            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def update(self, cove_dict, delta=None):
        # for layer_name in cove_dict.keys():
        #     is_covered = cove_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cove_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cove_dict_new):
        new_rate = self.all_coverage(cove_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)



class LayerTopKPattern(object):
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.layer_pattern = {}
        self.network_pattern = set()
        self.current = 0
        for (layer_name, layer_size) in layer_size_dict.items():
            self.layer_pattern[layer_name] = set()

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cove_dict = self.calculate(data)
        self.update(cove_dict)

    def calculate(self, data):
        layer_pat = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        topk_idx_list = []
        for (layer_name, layer_output) in layer_output_dict.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            # idx: (batch_size, k)
            pat = set([str(s) for s in list(idx[:, ])])
            topk_idx_list.append(idx)
            layer_pat[layer_name] = set.union(pat, self.layer_pattern[layer_name])
        network_topk_idx = torch.cat(topk_idx_list, 1)
        network_pat = set([str(s) for s in list(network_topk_idx[:, ])])
        network_pat = set.union(network_pat, self.network_pattern)
        return {
            'layer_pat': layer_pat,
            'network_pat': network_pat
        }

    def update(self, all_pat_dict, delta=None):
        layer_pat = all_pat_dict['layer_pat']
        network_pat = all_pat_dict['network_pat']
        self.layer_pattern = layer_pat
        self.network_pattern = network_pat
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_pat_dict)

    def coverage(self, pattern):
        return len(pattern)

    def all_coverage(self, all_pat_dict):
        network_pat = all_pat_dict['network_pat']
        return len(network_pat)

    def gain(self, all_pat_dict):
        new_rate = self.all_coverage(all_pat_dict)
        return new_rate - self.current

    def save(self, path):
        pass

    def load(self, path):
        pass

class LayerTFC(object):
    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.distant_dict = {}
        self.flann_dict = {}

        for (layer_name, layer_size) in layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

        self.current = 0

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        dis_dict = self.calculate(data)
        self.update(dis_dict)

    def update(self, dis_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dis_dict[layer_name]
            self.flann_dict[layer_name].build_index(np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.all_coverage(dis_dict)

    def calculate(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        dis_dict = {}
        for (layer_name, layer_output) in layer_output_dict.items():
            dis_dict[layer_name] = []
            for single_output in layer_output:
                single_output = single_output.cpu().numpy()
                if len(self.distant_dict[layer_name]) > 0:
                    _, approx_distances = self.flann_dict[layer_name].nn_index(np.expand_dims(single_output, 0), num_neighbors=1)
                    exact_distances = [
                        np.sum(np.square(single_output - distant_vec))
                        for distant_vec in self.distant_dict[layer_name]
                    ]
                    buffer_distances = [
                        np.sum(np.square(single_output - buffer_vec))
                        for buffer_vec in dis_dict[layer_name]
                    ]
                    nearest_distance = min(exact_distances + approx_distances.tolist() + buffer_distances)
                    if nearest_distance > self.threshold:
                        dis_dict[layer_name].append(single_output)
                else:
                    self.flann_dict[layer_name].build_index(single_output)
                    self.distant_dict[layer_name].append(single_output)
        return dis_dict

    def coverage(self, distant):
        return len(distant)

    def all_coverage(self, dis_dict):
        total = 0
        for layer_name in dis_dict.keys():
            total += len(dis_dict[layer_name])
        return total

    def gain(self, dis_dict):
        increased = self.all_coverage(dis_dict)
        return increased

    def save(self, path):
        pass

    def load(self, path):
        pass

if __name__ == '__main__':
    pass

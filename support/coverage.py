import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset
import numpy as np

from tqdm import tqdm
from pyflann import FLANN

from . import tool


def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    # out_max = out.max()
    # out_min = out.min()
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

class LayerNCS(object):  # S stands for "scaled"
    def __init__(self, model, threshold, layer_size_dict):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            # self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).cuda().type(torch.cuda.BoolTensor)
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0], dtype=torch.bool)
        self.current = 0

    def build(self, data_list):
        """Uses the given list of data to update the current coverage state."""
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            # data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cov_dict = self.calculate(data)
        self.update(cov_dict)

    def calculate(self, data):
        """Given the input data, returns each layer's coverage as if the new
        data is in the dataset upon which initial coverage is built.
        This function is pure."""
        cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cov_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cov_dict

    def update(self, cov_dict, delta=None):
        """Given a cov_dict, sets it as the current state and also updates the
        recorded overall coverage value with it.
        If delta is given, uses it directly to update the overall coverage."""
        # for layer_name in cov_dict.keys():
        #     is_covered = cov_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cov_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cov_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cov_dict):
        """Given a cov_dict, returns the overall coverage. Pure function."""
        (cove, total) = (0, 0)
        for layer_name in cov_dict.keys():
            is_covered = cov_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()  # ???

    def gain(self, cov_dict_new):
        """Given a coverage dict computed by calculate(), returns the overall
        difference between it and the current state."""
        new_rate = self.all_coverage(cov_dict_new)
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
        cov_dict = self.calculate(data)
        self.update(cov_dict)

    def calculate(self, data):
        cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            mask_index = layer_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cov_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cov_dict

    def update(self, cov_dict, delta=None):
        # for layer_name in cov_dict.keys():
        #     is_covered = cov_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cov_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cov_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cov_dict):
        (cove, total) = (0, 0)
        for layer_name in cov_dict.keys():
            is_covered = cov_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cov_dict_new):
        new_rate = self.all_coverage(cov_dict_new)
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
            'multisec_cov_dict': self.coverage_multisec_dict
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
        all_cov_dict = self.calculate(data)
        self.update(all_cov_dict)

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
        multisec_cov_dict = {}
        lower_cov_dict = {}
        upper_cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(num_neuron, self.k + 1).cuda().type(torch.cuda.BoolTensor)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            # (layer_output - l_bound) / div is between 0, 1
            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.cuda.LongTensor) * multisec_index
            # index 0 indicates invalid output (caused by multisec_index entry being 0)

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cov_dict[layer_name] = multisec_covered | self.coverage_multisec_dict[layer_name]

        return {
            'multisec_cov_dict': multisec_cov_dict
        }

    def update(self, all_cov_dict, delta=None):
        for k in all_cov_dict.keys():
            self.coverage_dict[k] = all_cov_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cov_dict)

    def coverage(self, all_covered_dict):
        multisec_covered = all_covered_dict['multisec_covered']

        num_neuron = multisec_covered.size(0)
        multisec_num_covered = torch.sum(multisec_covered[:, 1:])
        multisec_rate = multisec_num_covered / (num_neuron * self.k)

        return multisec_rate.item()

    def all_coverage(self, all_cov_dict):
        multisec_cov_dict = all_cov_dict['multisec_cov_dict']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cov_dict.keys():
            multisec_covered = multisec_cov_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total
        return multisec_rate.item()

    def gain(self, cov_dict_new):
        new_rate = self.all_coverage(cov_dict_new)
        return new_rate - self.current

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


class LayerSNA(object):  # Strong Neuron Activation Coverage... Subset of NBC {{{
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
            'upper_cov_dict': self.coverage_upper_dict
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
        all_cov_dict = self.calculate(data)
        self.update(all_cov_dict)

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
        upper_cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cov_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'upper_cov_dict': upper_cov_dict
        }

    def update(self, all_cov_dict, delta=None):
        for k in all_cov_dict.keys():
            self.coverage_dict[k] = all_cov_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cov_dict)

    def coverage(self, all_covered_dict):
        upper_covered = all_covered_dict['upper_covered']
        upper_rate = upper_covered.sum() / len(upper_covered)
        return upper_rate.item()

    def all_coverage(self, all_cov_dict):
        upper_cov_dict = all_cov_dict['upper_cov_dict']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cov_dict.keys():
            upper_covered = upper_cov_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()

    def gain(self, cov_dict_new):
        new_rate = self.all_coverage(cov_dict_new)
        return new_rate - self.current

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
# }}}

# Multi-stage coverage model. Needs to be built first during training.
class LayerNBC(object):  # Neuron Boundary Coverage
    def __init__(self, model, k, layer_size_dict):
        self.model = model
        self.k = k
        self.range_dict = {}
        self.coverage_lower_dict = {}
        self.coverage_upper_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            num_neuron = layer_size[0]
            # self.coverage_lower_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            # self.coverage_upper_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.coverage_lower_dict[layer_name] = torch.zeros(num_neuron, dtype=torch.bool)
            self.coverage_upper_dict[layer_name] = torch.zeros(num_neuron, dtype=torch.bool)
            # self.range_dict[layer_name] = [torch.ones(num_neuron).cuda() * 10000, torch.ones(num_neuron).cuda() * -10000]
            self.range_dict[layer_name] = [torch.ones(num_neuron) * 10000, torch.ones(num_neuron) * -10000]
        self.coverage_dict = {
            'lower_cov_dict': self.coverage_lower_dict,
            'upper_cov_dict': self.coverage_upper_dict
        }
        self.current = 0

    def build(self, data_list):
        print('Building Range...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            # data = data.cuda()
            self.set_range(data)
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            # data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cov_dict = self.calculate(data)
        self.update(all_cov_dict)

    def set_range(self, data):
        # Updates the upper and lower bounds for each neuron in each layer in self.range_dict
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        lower_cov_dict = {}
        upper_cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]

            # Only makes sense with testing data (i.e. not training data)
            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cov_dict[layer_name] = lower_covered | self.coverage_lower_dict[layer_name]
            upper_cov_dict[layer_name] = upper_covered | self.coverage_upper_dict[layer_name]

        return {
            'lower_cove_dict': lower_cov_dict,
            'upper_cove_dict': upper_cov_dict
        }

    def update(self, all_cov_dict, delta=None):
        for k in all_cov_dict.keys():
            self.coverage_dict[k] = all_cov_dict[k]
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(all_cov_dict)

    def coverage(self, all_covered_dict):
        lower_covered = all_covered_dict['lower_covered']
        upper_covered = all_covered_dict['upper_covered']

        lower_rate = lower_covered.sum() / len(lower_covered)
        upper_rate = upper_covered.sum() / len(upper_covered)

        return (lower_rate + upper_rate).item() / 2

    def all_coverage(self, all_cov_dict):
        lower_cov_dict = all_cov_dict['lower_cove_dict']
        upper_cov_dict = all_cov_dict['upper_cove_dict']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cov_dict.keys():
            lower_covered = lower_cov_dict[layer_name]
            upper_covered = upper_cov_dict[layer_name]

            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()

            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = lower_cove / lower_total
        upper_rate = upper_cove / upper_total
        return (lower_rate + upper_rate).item() / 2

    def gain(self, cov_dict_new):
        new_rate = self.all_coverage(cov_dict_new)
        return new_rate - self.current

    def save(self, path):
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=torch.device('cpu'))
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
            # self.coverage_dict[layer_name] = torch.zeros(num_neuron).cuda().type(torch.cuda.BoolTensor)
            self.coverage_dict[layer_name] = torch.zeros(num_neuron, dtype=torch.bool)
        self.current = 0

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            # data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        cov_dict = self.calculate(data)
        self.update(cov_dict)

    def calculate(self, data):
        cov_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            # covered = torch.zeros(layer_output.size()).cuda()
            covered = torch.zeros(layer_output.size())
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1

            is_covered = covered.sum(0) > 0
            cov_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cov_dict

    def update(self, cov_dict, delta=None):
        # for layer_name in cov_dict.keys():
        #     is_covered = cov_dict[layer_name]
        #     self.coverage_dict[layer_name] = is_covered
        self.coverage_dict = cov_dict
        if delta:
            self.current += delta
        else:
            self.current = self.all_coverage(cov_dict)

    def coverage(self, is_covered):
        rate = is_covered.sum() / len(is_covered)
        return rate

    def all_coverage(self, cov_dict):
        (cove, total) = (0, 0)
        for layer_name in cov_dict.keys():
            is_covered = cov_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def gain(self, cov_dict_new):
        new_rate = self.all_coverage(cov_dict_new)
        return new_rate - self.current

    def save(self, path):
        torch.save(self.coverage_dict, path)

    def load(self, path):
        self.coverage_dict = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Loaded coverage: %f' % loaded_cov)



class LayerTopKPattern(object):  # Mainly post-processing {{{
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
        cov_dict = self.calculate(data)
        self.update(cov_dict)

    def calculate(self, data):
        layer_pat = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        topk_idx_list = []
        for (layer_name, layer_output) in layer_output_dict.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            # idx: (batch_size, k)
            pat = set([str(s) for s in list(idx[:, ])])
            # pat is a list of stringified per-input top-k indices
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
        # Works by counting number of different patterns
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
# }}}

class LayerTFC(object):  # Needs FLANN {{{
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
# }}}

class Estimator(object):
    def __init__(self, feature_num, class_num=1, use_cuda=True):
        self.device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.class_num = class_num
        # set it to 1; class-conditional covariance is too costly
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(class_num, feature_num).to(self.device)
        self.Amount = torch.zeros(class_num).to(self.device)

    def calculate(self, features, labels=None):
        # This an incrementally updated covariance.
        # If we don't want class-conditional covar here,
        # simply set `labels` to a vector zeros.
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        if not labels:
            labels = torch.zeros(N, dtype=torch.int64).to(self.device)

        # Expand to prepare for class-conditional covar
        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        # Mark the label for each sample, NxC
        # All-one vector if non-class-conditional
        onehot.scatter_(1, labels.view(-1, 1), 1)

        # Each row (last dim) is filled with the same element (0 or 1)
        # All-one array when non-class-conditional
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        # mul is element-wise multiplication so this is just masking.
        # For each sample, preserves only feature rows within enabled classes.
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        # Sum over the batch dimension to count the times each class appears
        # Each row still has the same element (the count)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1  # Avoid division by zero

        # Average of each feature in each class
        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),  # CxAxN
            var_temp.permute(1, 0, 2)  # CxNxA
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        # For each class, make a matrix filled with the same element (the count)
        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) +
                          var_temp.mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave, 
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, covar_dict):
        self.Ave = covar_dict['Ave']
        self.CoVariance = covar_dict['CoVariance']
        self.Amount = covar_dict['Amount']

class LayerCovar(object):
    def __init__(self, model, layer_size_dict, topn_sing_vals=10, linear_layers_only=False):
        self.model = model
        self.topn_sing_vals = topn_sing_vals
        self.covar_dict = {}
        self.sing_val_dict = {}
        for (layer_name, layer_size) in layer_size_dict.items():
            if linear_layers_only and 'Linear' not in layer_name:
                continue
            num_neuron = layer_size[0]
            self.covar_dict[layer_name] = Estimator(num_neuron)
            self.sing_val_dict[layer_name] = torch.tensor([0.0] * topn_sing_vals).to(self.covar_dict[layer_name].device)

    def build(self, data_list):
        print('Building Coverage...')
        for i in tqdm(range(len(data_list))):
            data = data_list[i]
            # data = data.cuda()
            self.build_step(data)

    def build_step(self, data):
        all_cov_dict = self.calculate(data)
        self.update(all_cov_dict)

    def calculate(self, data):
        ret = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            if layer_name not in self.covar_dict:
                continue
            ret[layer_name] = self.covar_dict[layer_name].calculate(layer_output)
        return ret

    def update(self, all_cov_dict, delta=None):
        for k in self.covar_dict.keys():
            self.covar_dict[k].update(all_cov_dict[k])
            self.sing_val_dict[k] = torch.linalg.svdvals(self.covar_dict[k].CoVariance)[:self.topn_sing_vals]
        if delta:
            raise NotImplementedError

    def all_coverage(self, all_cov_dict):
        # Doesn't really have this concept
        raise NotImplementedError

    def gain(self, cov_dict_new):
        gain = 0
        for lname, cov_dict in cov_dict_new.items():
            if lname not in self.covar_dict:
                continue
            new_covar = cov_dict['CoVariance']
            # Algorithm: SVD
            # new_sing_vals = torch.linalg.svdvals(new_covar)[:self.topn_sing_vals]
            # old_sing_vals = self.sing_val_dict[lname]
            # gain += torch.sum(torch.maximum(new_sing_vals - old_sing_vals, torch.Tensor([0])))
            # Algorithm: L1 Norm
            gain += torch.sum(torch.abs(new_covar - self.covar_dict[lname].CoVariance))
            # Algorithm: L2 Norm
            # gain += torch.sum((new_covar - self.covar_dict[lname].CoVariance) ** 2)
        return gain

    def save(self, path):
        state = {
            'covar': {
                layer_name: {
                    'Ave': self.covar_dict[layer_name].Ave,
                    'CoVariance': self.covar_dict[layer_name].CoVariance,
                    'Amount': self.covar_dict[layer_name].Amount
                } for layer_name in self.covar_dict.keys()
            },
            'sing_val': {
                layer_name: self.sing_val_dict[layer_name]
                for layer_name in self.sing_val_dict.keys()
            }
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        for layer_name in self.covar_dict.keys():
            self.covar_dict[layer_name].update(state['covar'][layer_name])
            self.sing_val_dict[layer_name] = state['sing_val'][layer_name]

        # loaded_cov = self.all_coverage(self.coverage_dict)
        # print('Loaded coverage: %f' % loaded_cov)


# vim: set fdm=marker:

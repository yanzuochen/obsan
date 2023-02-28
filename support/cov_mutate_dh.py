import sys
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from my_utils import save_image
from estimator import Estimator
from criteria import *
from style_operator import Stylized

class Parameters(object):
    def __init__(self, base_args):
        import itertools
        import image_transforms
        # self.exp_name = 'imagenet-nc' # FIXME
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.metric = base_args.metric
        self.class_cond = base_args.class_cond
        self.num_workers = 4
        self.NC_threshold = -1
        self.TFC_threshold = -1
        self.KMN_K = -1
        self.TopK_K = -1

        self.batch_size = 50
        self.mutate_batch_size = 1
        self.nc = 3
        self.image_size = 128 if self.dataset == 'ImageNet' else 32
        self.input_shape = (1, self.image_size, self.image_size, 3)
        self.num_cat = 100 if self.dataset == 'ImageNet' else 10
        self.num_perclass = 1000 // self.num_cat

        self.seed_num = 1000

        self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16
        # self.p_min = 0.01
        # self.gamma = 5
        self.alpha = 0.2 # default 0.02
        self.beta = 0.5 # default 0.2
        self.TRY_NUM = 50
        self.save_every = 100
        self.output_root = '/data/yyuanaq/output/Coverage/Fuzzer/'
        self.use_stat = self.metric == 'Stat'

        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                            [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))
        # rotation = list(
        #     itertools.product([getattr(image_transforms, "image_rotation")], [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]))

        # translation = list(itertools.product([getattr(image_transforms, "image_translation")], list(xrange(-50, 50))))
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        # shear = list(itertools.product([getattr(image_transforms, "image_shear")], list(range(-3, 3))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation #+ shear
        self.P = contrast + brightness + blur
        self.S = list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        self.save_batch = False

# class INFO(object):
#     def __init__(self):
#         self.dict = {}

#     def __getitem__(self, i):
#         _i = str(i)
#         if _i in self.dict:
#             return self.dict[_i]
#         else:
#             I0, I0_new, state = np.copy(i), np.copy(i), 0
#             return I0, I0_new, state

#     def __setitem__(self, i, s):
#         _i = str(i)
#         self.dict[_i] = s
#         return self.dict[_i]

class INFO(object):
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, state = i, 0
            return I0, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]

class Fuzzer(object):
    def __init__(self, params, coverage):
        self.params = params
        self.epoch = 0
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.info = INFO()
        self.hyper_params = {
            'alpha': 0.4, # [0, 1], default 0.02, 0.1 # number of pix
            'beta': 0.8, # [0, 1], default 0.2, 0.5 # max abs pix
            'TRY_NUM': 50,
            'p_min': 0.01,
            'gamma': 5,
            'K': 64
        }
        self.logger = my_utils.Logger(params, self)
        self.coverage = coverage
        self.initial_coverage = copy.deepcopy(coverage.current)
        self.delta_time = 0
        self.delta_batch = 0
        self.num_ae = 0
        self.cov_operator_dict = {}
        self.ae_operator_dict = {}

    def exit(self):
        self.print_info()
        self.coverage.save(self.params.coverage_root + 'coverage.pth')
        self.logger.save()
        self.logger.exit()
        self.save_op_info()

    def save_op_info(self):
        with open(self.params.coverage_root + 'cov_op_cnt.json', 'a') as f:
            json.dump(self.cov_operator_dict, f)
        with open(self.params.coverage_root + 'ae_op_cnt.json', 'a') as f:
            json.dump(self.ae_operator_dict, f)

    def update_cov_op(self, op):
        t, p = op
        if t.__name__ in self.cov_operator_dict.keys():
            self.cov_operator_dict[t.__name__] += 1
        else:
            self.cov_operator_dict[t.__name__] = 1

    def update_ae_op(self, op):
        t, p = op
        if t.__name__ in self.ae_operator_dict.keys():
            self.ae_operator_dict[t.__name__] += 1
        else:
            self.ae_operator_dict[t.__name__] = 1

    def can_terminate(self):
        condition = sum([
            self.epoch > 10000,
            self.delta_time > 60 * 60 * 6,
            # self.delta_batch > 10000
        ])
        return condition > 0

    def print_info(self):
        self.logger.update(self)

    def save_ae(self, ae_image, ae_label, path):
        with torch.no_grad():
            data = {
                'image': ae_image,
                'label': ae_label
            }
            torch.save(data, path)
            # print('AE saved on %s' % path)

    def save_gen(self, image, label, path):
        with torch.no_grad():
            data = {
                'image': image,
                'label': label
            }
            torch.save(data, path)


    def is_adversarial(self, image, label, k=1):
        with torch.no_grad():
            # image = torch.from_numpy(image_numpy)
            # label = torch.from_numpy(label_numpy)
            scores = self.coverage.model(image)
            _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
            correct = ind.eq(label.view(-1, 1).expand_as(ind))
            wrong = ~correct
            index = (wrong == True).nonzero(as_tuple=True)[0]
            wrong_total = wrong.view(-1).float().sum()
            if wrong_total > 0:
                self.logger.count(label.view(-1, 1).squeeze(1), ind.view(-1, 1).squeeze(1))
            return wrong_total, index

    def to_batch(self, data_list):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.params.mutate_batch_size == 0:
                batch_list.append(np.stack(batch, 0))
                batch = []
            batch.append(data)
        if len(batch):
            batch_list.append(np.stack(batch, 0))
        return batch_list

    def image_to_input(self, image):
        scaled_image = image / self.params.input_scale
        tensor_image = torch.from_numpy(scaled_image).transpose(1, 3)
        normalized_image = my_utils.image_normalize(tensor_image, self.params.dataset)
        return normalized_image

    def run(self, I_input, L_input):
        F = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
        T = self.Preprocess(I_input, L_input)

        del I_input
        del L_input
        gc.collect()

        B, B_label, B_id = self.SelectNext(T)
        self.epoch = 0
        start_time = time.time()
        while not self.can_terminate():
            # if int(self.delta_time) // self.time_slot == self.time_idx:
            #     self.print_info()
            #     self.time_idx += 1
            if self.epoch % 500 == 0:
                self.print_info()
            # S = self.Sample(B)
            S = B
            S_label = B_label
            Ps = self.PowerSchedule(S, self.hyper_params['K'])
            B_new = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_old = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_label_new = []
            for s_i in range(len(S)):
                I = S[s_i]
                L = S_label[s_i]
                for i in range(1, Ps(s_i) + 1):
                    I_new, op = self.Mutate(I)
                    if self.isFailedTest(I_new):
                        F += np.concatenate((F, [I_new]))
                    elif self.isChanged(I, I_new):

                        torch_image = self.image_to_input(np.concatenate((B_new, [I_new])))
                        torch_image = torch_image.cuda()
                        torch_label = torch.from_numpy(np.array(B_label_new + [L]))
                        torch_label = torch_label.cuda()

                        if self.params.use_stat:
                            cove_dict = self.coverage.calculate(torch_image, torch_label)
                        else:
                            cove_dict = self.coverage.calculate(torch_image)
                        delta = self.coverage.gain(cove_dict)

                        if self.CoverageGain(delta):
                            self.coverage.update(cove_dict, delta)
                            self.update_cov_op(op)
                            B_new = np.concatenate((B_new, [I_new]))
                            B_old = np.concatenate((B_old, [I]))
                            B_label_new += [L]
                            break


            if len(B_new) > 0:
                B_label_new = np.array(B_label_new)
                # scaled_image = B_new / self.params.input_scale
                # new_image = torch.from_numpy(scaled_image).transpose(1, 3)
                new_image = self.image_to_input(B_new)
                new_image = new_image.cuda()
                new_label = torch.from_numpy(B_label_new)
                new_label = new_label.cuda()

                # if self.params.use_stat:
                #     cove_dict = self.coverage.calculate(new_image, new_label)
                # else:
                #     cove_dict = self.coverage.calculate(new_image)
                # delta = self.coverage.gain(cove_dict)

                # if self.CoverageGain(delta):
                if True:
                    # self.update_cov_op(op)
                    # print("coverage:", self.coverage.current)
                    # self.coverage.update(cove_dict, delta)
                    B_c, Bs, Bs_label = T
                    B_c += [0]
                    Bs += [B_new]
                    Bs_label += [B_label_new]
                    self.delta_batch += 1
                    self.BatchPrioritize(T, B_id)

                    self.save_gen(new_image, new_label, self.params.gen_root + ('%07d_gen.pth' % self.epoch))

                    num_wrong, ae_index = self.is_adversarial(new_image, new_label)
                    if num_wrong > 0:
                        self.update_ae_op(op)
                        self.num_ae += num_wrong
                        self.save_ae(new_image[ae_index], new_label[ae_index], self.params.ae_root + ('%07d_ae.pth' % self.epoch))

                    if self.epoch % self.params.save_every == 0:
                        self.saveImage(B_new / self.params.input_scale, self.params.image_root + ('%03d_new.jpg' % self.epoch))
                        self.saveImage(B_old / self.params.input_scale, self.params.image_root + ('%03d_old.jpg' % self.epoch))
                        if num_wrong > 0:
                            save_image(new_image[ae_index].data, self.params.image_root + ('%03d_ae.jpg' % self.epoch), normalize=True)
                # if num_wrong > 0:
                #     save_image(new_image[ae_index].data, self.params.ae_root + ('%03d_ae.jpg' % self.epoch), normalize=True)

            gc.collect()

            B, B_label, B_id = self.SelectNext(T)
            self.epoch += 1
            self.delta_time = time.time() - start_time


    def Preprocess(self, image_list, label_list):
        # _I = np.random.permutation(I * self.params.input_scale)
        # Bs = np.array_split(_I, range(self.params.batch1, len(_I), self.params.batch1))

        randomize_idx = np.arange(len(image_list))
        np.random.shuffle(randomize_idx)

        # max_p1 = 0
        # for image in image_list:
        #     max_p1 = max(max_p1, np.max(image))
        # print('max_p1: ', max_p1)

        image_list = [image_list[idx] * self.params.input_scale for idx in randomize_idx]

        # max_p2 = 0
        # for image in image_list:
        #     max_p2 = max(max_p2, np.max(image))
        # print('max_p2: ', max_p2)
        label_list = [label_list[idx] for idx in randomize_idx]

        Bs = self.to_batch(image_list)
        Bs_label = self.to_batch(label_list)
        # print('Bs: ', len(Bs))
        # print('Bs[0]: ', Bs[0].shape)
        return list(np.zeros(len(Bs))), Bs, Bs_label


    def calc_priority(self, B_ci):
        if B_ci < (1 - self.hyper_params['p_min']) * self.hyper_params['gamma']:
            return 1 - B_ci / self.hyper_params['gamma']
        else:
            return self.hyper_params['p_min']

    def SelectNext(self, T):
        B_c, Bs, Bs_label = T
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(Bs), p=B_p / np.sum(B_p))
        return Bs[c], Bs_label[c], c


    def Sample(self, B):
        # c = np.random.choice(len(B), size=self.params.batch2, replace=False)
        c = np.random.choice(len(B), size=self.params.mutate_batch_size, replace=False)
        return B[c]


    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i]
            I0, state = self.info[I]
            p = self.hyper_params['beta'] * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))

        return Ps


    def isFailedTest(self, I_new):
        return False


    def isChanged(self, I, I_new):
        return np.any(I != I_new)


    def CoverageGain(self, cov):
        if self.params.metric == 'Stat':
            return cov is not None
        else:
            return cov > 0


    def BatchPrioritize(self, T, B_id):
        B_c, Bs, Bs_label = T
        B_c[B_id] += 1


    def Mutate(self, I):
        G, P, S = self.params.G, self.params.P, self.params.S
        I0, state = self.info[I]

        for i in range(1, self.hyper_params['TRY_NUM']):
            if state == 0:
                t, p = self.randomPick(G + P + S)
            else:
                t, p = self.randomPick(P + S)

            I_mutated = t(I, p).reshape(*(self.params.input_shape[1:]))
            I_mutated = np.clip(I_mutated, 0, 255)

            if (t, p) in S or self.f(I0, I_mutated):
                if (t, p) in G:
                    state = 1
                    I0_G = t(I0, p)
                    I0_G = np.clip(I0_G, 0, 255)
                    self.info[I_mutated] = (I0_G, state)
                else:
                    self.info[I_mutated] = (I0, state)
                # print('I0_G', np.max(I0_G))
                # print('I_mutated', np.max(I_mutated))
                return I_mutated, (t, p)
        return I, (t, p)


    # def Mutate(self, I):
    #     G, P, S = self.params.G, self.params.P, self.params.S

    #     for i in range(1, self.hyper_params['TRY_NUM']):
    #         t, p = self.randomPick(G + P + S)

    #         I_mutated = t(I, p).reshape(*(self.params.input_shape[1:]))
    #         I_mutated = np.clip(I_mutated, 0, 255)

    #         if (t, p) in S or \
    #            (t, p) in G or \
    #            self.f(I, I_mutated):
    #             # print('I0_G', np.max(I0_G))
    #             # print('I_mutated', np.max(I_mutated))
    #             return I_mutated, (t, p)

    #     return I, (t, p)

    def saveImage(self, image, path):
        if not image is None:
            image_tensor = torch.from_numpy(image).transpose(1, 3)
            save_image(image_tensor.data, path, normalize=True)
            print('Image saved on %s' % path)

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]


    def f(self, I, I_new):
        if (np.sum((I - I_new) != 0) < self.hyper_params['alpha'] * np.sum(I > 0)):
            return np.max(np.abs(I - I_new)) <= 255
        else:
            return np.max(np.abs(I - I_new)) <= self.hyper_params['beta'] * 255


if __name__ == '__main__':
    import os
    import argparse
    import torchvision
    import gc

    from dataset import CIFAR10Dataset, ImageNetDatasetCov
    import neuron_coverage as tool
    import my_utils
    import classifier as cifar10_models

    import signal
    def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            try:
                if engine is not None:
                    engine.print_info()
                    engine.save_op_info()
                    if engine.logger is not None:
                        engine.logger.save()
                        engine.logger.exit()
                    if engine.coverage is not None:
                        engine.coverage.save(args.coverage_root + 'coverage_int.pth')
            except:
                pass
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    os.environ['TORCH_HOME'] = '/data/yyuanaq/collection/ImageNet/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                            choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'vgg16_bn', 'mobilenet_v2', 'googlenet', 'inception_v3'])
    parser.add_argument('--metric', type=str, default='NC',
                            choices=['Stat', 'NC', 'NCS', 'KMN', 'SNA', 'NBC', 'TopK', 'TopKPatt', 'TFC'])
    parser.add_argument('--class_cond', type=int, default=1, choices=[0, 1])
    base_args = parser.parse_args()

    args = Parameters(base_args)

    args.exp_name = ('%s-%s-%s' % (args.dataset, args.model, args.metric))
    print(args.exp_name)
    my_utils.make_path(args.output_root)
    my_utils.make_path(args.output_root + args.exp_name)

    args.image_root = args.output_root + args.exp_name + '/image/'
    args.ae_root = args.output_root + args.exp_name + '/ae/'
    args.gen_root = args.output_root + args.exp_name + '/gen/'
    args.coverage_root = args.output_root + args.exp_name + '/coverage/'
    args.log_root = args.output_root + args.exp_name + '/log/'

    my_utils.make_path(args.image_root)
    my_utils.make_path(args.ae_root)
    my_utils.make_path(args.gen_root)
    my_utils.make_path(args.coverage_root)
    my_utils.make_path(args.log_root)

    if args.dataset == 'ImageNet':
        model = torchvision.models.__dict__[args.model](pretrained=False)
        path = ('/data/yyuanaq/collection/ImageNet/%s.pth' % args.model)
        model.eval()
        model.load_state_dict(torch.load(path))
        args.z_dim = 120
        args.image_size = 128
        # args.num_cat = 100
        # assert args.num_cat <= 1000
    elif args.dataset == 'CIFAR10':
        model = getattr(cifar10_models, args.model)(pretrained=False)
        path = ('/data/yyuanaq/collection/CIFAR-10/my/cifar10/%s/%s.pt' % (args.model, args.model))
        model.eval()
        model.load_state_dict(torch.load(path))
        args.z_dim = 128
        args.image_size = 32
        # args.num_cat = 10
        # assert args.num_cat <= 10

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).cuda()
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)

    # seed = 233
    # data_set = NoiseDataset(args, seed)
    # latent_list, label_list = data_set.build()
    # image_batch_list = data_set.batch_gen(latent_list, label_list)
    # image_numpy_list = data_set.to_numpy(image_batch_list)

    if args.dataset == 'CIFAR10':
        data_set = CIFAR10Dataset(args, split='test')
    elif args.dataset == 'ImageNet':
        data_set = ImageNetDatasetCov(args, split='val')

    image_list, label_list = data_set.build()
    if not args.class_cond:
        label_list = [torch.LongTensor([0]).squeeze() for _ in range(len(image_list))]
    image_batch_list = data_set.to_batch(image_list)
    image_numpy_list = data_set.to_numpy(image_list)
    label_batch_list = data_set.to_batch(label_list, False)
    label_numpy_list = data_set.to_numpy(label_list, False)

    del image_list
    del label_list
    gc.collect()

    CoveDict = {
        'Stat': LayerStat,
        'NC': LayerNC,
        'NCS': LayerNCS,
        'KMN': LayerKMN,
        'SNA': LayerSNA,
        'NBC': LayerNBC,
        'TopK': LayerTopK,
        'TopKPatt': LayerTopKPattern,
        'TFC': LayerTFC
    }
    ParamDict = {
        'Stat': args.num_cat if args.class_cond else 1,
        'NC': 0,
        'NCS': 0.75,
        'KMN': 100,
        'SNA': 1,
        'NBC': 1,
        'TopK': 10,
        'TopKPatt': 50,
        'TFC': 10 if args.dataset == 'CIFAR10' else 1000
    }

    coverage = CoveDict[args.metric](model, ParamDict[args.metric], layer_size_dict)


    if args.metric in ['KMN', 'NBC', 'SNA']:
        from dataset import TorchCIFAR10Dataset, TorchImageNetDatasetCov

        if args.dataset == 'CIFAR10':
            train_data = TorchCIFAR10Dataset(args, split='train')
        elif args.dataset == 'ImageNet':
            train_data = TorchImageNetDatasetCov(args, split='train')
        train_loader = torch.utils.data.DataLoader(
                            train_data,
                            batch_size=200,
                            num_workers=4,
                            shuffle=False
                        )
        print('Building range...')
        for i, (image, label) in enumerate(tqdm(train_loader)):
            image = image.cuda()
            coverage.set_range(image)
        print('Initializing...')
        for image in tqdm(image_batch_list):
            image = image.cuda()
            coverage.build_step(image)
    else:
        if args.use_stat:
            coverage.build(image_batch_list, label_batch_list)
        else:
            coverage.build(image_batch_list)

    del image_batch_list
    del label_batch_list
    gc.collect()

    initial_coverage = copy.deepcopy(coverage.current)
    print('Initial Coverage: %f' % initial_coverage)
    engine = Fuzzer(args, coverage)

    engine.run(image_numpy_list, label_numpy_list)
    # engine.print_info()
    # coverage.save(args.coverage_root + 'coverage.pth')
    engine.exit()


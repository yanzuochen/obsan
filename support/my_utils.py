import torch
import math
irange = range
import os
import random
import numpy as np
import shutil
import json
# import progressbar
import torchvision
import torchvision.transforms as transforms
# from scipy.stats import truncnorm
#import fastBPE

#import TransCoder.preprocessing.src.code_tokenizer as code_tokenizer
#from TransCoder.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD

def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
       transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise NotImplementedError
    return transform(image)

def image_normalize_inv(image, dataset):
    if dataset == 'CIFAR10':
        transform = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = NormalizeInverse((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise NotImplementedError
    return transform(image)

class Logger(object):
    def __init__(self, args, engine):
        import time
        self.name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log'
        self.args = args
        self.log_path = os.path.join(args.log_root, self.name)
        if args.dataset == 'ImageNet':
            self.label2pred = torch.zeros(args.num_cat, 1000)
        elif args.dataset == 'CIFAR10':
            self.label2pred = torch.zeros(args.num_cat, 10)
        self.f = open(self.log_path, 'a')
        self.f.write('Dataset: %s\n' % args.dataset)
        self.f.write('Model: %s\n' % args.model)
        self.f.write('Class: %d\n' % args.num_cat)
        self.f.write('Data in each class: %d\n' % args.num_perclass)
        self.f.write('Metric: %s\n' % args.metric)
        self.f.write('Blind: %s\n' % args.blind)
        for k in engine.hyper_params.keys():
            self.f.write('%s %s\n' % (k, engine.hyper_params[k]))
        self.f.flush()

    def update(self, engine):
        print('Epoch: %d' % engine.epoch)
        print('Delta coverage: %f' % (engine.coverage.current - engine.initial_coverage))
        print('Delta time: %fs' % engine.delta_time)
        print('Delta batch: %d' % engine.delta_batch)
        print('AE: %d' % engine.num_ae)
        self.f.write('Delta time: %fs, Epoch: %d, Current coverage: %f, Delta coverage:%f, AE: %d, Delta batch: %d\n' % \
            (engine.delta_time, engine.epoch, engine.coverage.current, \
             engine.coverage.current - engine.initial_coverage,
             engine.num_ae, engine.delta_batch))
        self.f.flush()

    def count(self, label, pred):
        index = tuple([label, pred])
        self.label2pred[index] += 1

    def save(self):
        path = os.path.join('/'.join(self.log_path.split('/')[:-1]), 'label2pred.pth')
        torch.save(self.label2pred, path)

    def exit(self):
        self.f.close()

def truncate(tensor, thresh):
    thresh_a = -thresh
    thresh_b = thresh
    return torch.clamp(tensor, thresh_a, thresh_b)

def truncated_normal_(tensor, mean=0, std=1, thresh_a=-2, thresh_b=2):
    assert thresh_a < thresh_b
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < thresh_b) & (tmp > thresh_a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GradSaver:
    def __init__(self):
        self.grad = -1

    def save_grad(self, grad):
        self.grad = grad

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    #print('bs:', batch_size)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    #print('ind: ', ind.shape)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    #print('correct: ', correct.shape)
    #print(correct)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() / batch_size

# class PreprocessorCPP(object):
#     def __init__(self, model_path, BPE_path):
#         reloaded = torch.load(model_path, map_location='cpu')
#         reloaded_params = AttrDict(reloaded['params'])
#         self.dico = Dictionary(
#                 reloaded['dico_id2word'],
#                 reloaded['dico_word2id'],
#                 reloaded['dico_counts']
#                 )

#         self.bpe_model = fastBPE.fastBPE(os.path.abspath(BPE_path))

#         lang = 'cpp'
#         self.tokenizer = getattr(code_tokenizer, f'tokenize_{lang}')
#         self.detokenizer = getattr(code_tokenizer, f'detokenize_{lang}')

#         lang += '_sa'
#         lang_id = reloaded_params.lang2id[lang]

#     def word_to_index(self, w):
#         return self.dico.index(w)

#     def index_to_word(self, idx):
#         return self.dico[idx]

#     def preprocess(self, input_code):
#         tokens = [t for t in self.tokenizer(input_code)]
#         tokens = self.bpe_model.apply(tokens)
#         tokens = ['</s>'] + tokens + ['</s>']
#         out = [self.dico.index(w) for w in tokens]
#         return out
#         # out = torch.LongTensor([self.dico.index(w)
#         #                         for w in tokens])[:, None]

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)

def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        # std_inv = 1 / (std + 1e-7)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        #return super().__call__(tensor)

def my_scale(v, v_max, v_min, low=0, up=1):
    return (up - low) * (v - v_min) / max(1e-7, v_max - v_min) + low

def my_scale_inv(v, v_max, v_min, low=0, up=1):
    return (v - low) / (up - low) * max(1e-7, v_max - v_min) + v_min

def get_widgets():
    return ['Progress: ', progressbar.Percentage(), ' ',
            progressbar.Bar('#'), ' ', 'Count: ', progressbar.Counter(), ' ',
            progressbar.Timer(), ' ', progressbar.ETA()]

def reparameterize(mu, logvar):
    logvar = logvar.mul(0.5).exp_()
    eps = logvar.data.new(logvar.size()).normal_()
    return eps.mul(logvar).add_(mu)

def KLDLoss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0).cuda()

# def accuracy(preds, y):
#     preds = torch.argmax(preds, dim=1)
#     correct = (preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc

def build_vocab(vocab_path, token_list):
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            token_vocab = vocab['token']
            index_vocab = vocab['index']
    else:
        token_vocab = {}
        index_vocab = {}
        count = {}
        cur_index = 0
        for token in token_list:
            for t in token:
                if t in count.keys():
                    count[t] += 1
                    if count[t] == 10:
                        token_vocab[t] = cur_index
                        index_vocab[cur_index] = t
                        cur_index += 1
                else:
                    count[t] = 1
        for t in ['<UNK>', '<START>', '<END>']:
            token_vocab[t] = cur_index
            index_vocab[cur_index] = t
            cur_index += 1
        with open(vocab_path, 'w') as f:
            json.dump({'token': token_vocab,
                       'index': index_vocab}, f)
    return token_vocab, index_vocab

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr).convert('RGB')
    im.save(filename)

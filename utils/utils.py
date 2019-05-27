import copy
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_vocabulary(vocab, path):
    with open(path, 'wb') as fw:
        pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocabulary(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_best_model(model, path, filename="saved_best_model.pt"):
    out_path = os.path.abspath(os.path.join(path, filename))
    torch.save(model, out_path)


def load_best_model(path, filename="saved_best_model.pt"):
    out_path = os.path.abspath(os.path.join(path, filename))
    model = torch.load(out_path)
    return model


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))


# Direct c/p from Pytorch/BiLSTM Tutorial
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec, dim=-1):
    max_score, _ = torch.max(vec, dim=dim)
    max_score_broadcast = max_score.unsqueeze(dim)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def scheduled_annealing_strategy(epoch, max_epoch, max=1.0, min=0.01, gain=0.3):
    upper_alpha = max - min
    lower_alpha = (1 + torch.exp(gain * (epoch - (max_epoch // 2))))
    alpha = (upper_alpha / lower_alpha) + max
    return alpha


def clones(module, num_of_clones):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_clones)])


def subsequent_mask(size):
    # Mask out subsequent positions. It is to prevent positions from attenting to subsequent positions
    # For more detailed information:
    # The Annotated Transformer = https://nlp.seas.hardvard.edu/2018/04/03.attention.html
    sm = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(sm) == 0


def gelu(x):
    # Gaussian Error Linear Unit
    # Ref: https://github.com/pytorch/pytorch/issues/20464
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


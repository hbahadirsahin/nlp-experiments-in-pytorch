import copy
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn


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
    return torch.load(out_path)


def calculate_accuracy(predictions, ground_truths):
    correct = (torch.max(predictions, 1)[1].view(ground_truths.size()).data == ground_truths.data)
    return float(correct.sum()) / len(correct) * 100


def calculate_topk_accuracy(output, target, topk=(2,)):
    """
    Code copied/pasted from PyTorch Imagenet example: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

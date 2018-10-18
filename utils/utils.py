import math
import time
import torch
import os
import pickle


def save_vocabulary(vocab, path):
    with open(path, 'wb') as fw:
        pickle.dump(vocab, fw, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocabulary(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_best_model(model, path):
    filename = "saved_best_model.pt"
    out_path = os.path.abspath(os.path.join(path, filename))
    torch.save(model, out_path)


def load_best_model(path):
    filename = "saved_best_model.pt"
    out_path = os.path.abspath(os.path.join(path, filename))
    return torch.load(out_path)


def calculate_accuracy(predictions, ground_truths):
    correct = (torch.max(predictions, 1)[1].view(ground_truths.size()).data == ground_truths.data)
    return float(correct.sum()) / len(correct)


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

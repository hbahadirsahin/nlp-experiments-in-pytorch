import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.args = args

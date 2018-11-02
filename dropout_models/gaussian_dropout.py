import torch
import torch.nn as nn


class GaussianDropout(nn.Module):
    def __init__(self, prob):
        super(GaussianDropout, self).__init__()
        if 0 < prob <= 0.5:
            self.alpha = torch.Tensor([prob / (1.0 - prob)])
        else:
            prob = 0.5
            self.alpha = torch.Tensor([prob / (1.0 - prob)])

    def forward(self, x):
        if self.train():
            # Epsilon ~ N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1
            if x.is_cuda:
                epsilon = epsilon.cuda()
            # Local reparametrization trick: x_i = ^x_i * epsilon_i
            return x * epsilon
        else:
            return x

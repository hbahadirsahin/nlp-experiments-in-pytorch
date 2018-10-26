import torch
import torch.nn as nn
import torch.distributions as dist


class GaussianDropout(nn.Module):
    def __init__(self, prob):
        super(GaussianDropout, self).__init__()
        if prob == 1.0:
            self.alpha = torch.Tensor([1.0])
        else:
            self.alpha = torch.Tensor([prob / (1.0 - prob)])

    def forward(self, x):
        if self.train():
            # Epsilon ~ N(1, alpha)
            normal = dist.Normal(torch.tensor[1.0], self.alpha)
            epsilon = normal.sample(x.size())

            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Local reparametrization trick: x_i = ^x_i * epsilon_i
            return x * epsilon
        else:
            return x

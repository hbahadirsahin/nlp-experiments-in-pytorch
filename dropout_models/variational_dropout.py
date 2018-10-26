import torch
import torch.nn as nn
import torch.distributions as dist


class VariationalDropout(nn.Module):
    def __init__(self, prob, dimension=None):
        super(VariationalDropout, self).__init__()

        self.dimension = dimension
        if prob == 1.0:
            self.alpha = 1.0
        else:
            self.alpha = prob / (1 - prob)

        log_alpha = (torch.ones(dimension) * self.alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl_divergence(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        constant = 0.5

        alpha = self.log_alpha.exp()

        kl = -constant * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        return kl.mean()

    @staticmethod
    def clip(tensor, to):
        return torch.clamp(tensor, -to, to)

    def forward(self, x):
        if self.train():
            # Epsilon ~ N(0, 1)
            normal = dist.Normal(torch.tensor[0.0], torch.tensor[1.0])
            epsilon = normal.sample(x.size())
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Shring tensor values to range [-alpha, alpha]
            self.log_alpha.data = self.clip(self.log_alpha.data, self.alpha)
            alpha = self.log_alpha.exp()

            # Epsilon ~ N(1, alpha)
            epsilon *= alpha

            return x * epsilon
        else:
            return x

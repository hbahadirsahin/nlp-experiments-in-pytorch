import torch
import torch.nn as nn
from torch.autograd import Variable


class VariationalDropout(nn.Module):
    def __init__(self, prob, dimension=None):
        super(VariationalDropout, self).__init__()

        self.dimension = dimension

        alpha = 1.0
        if prob <= 0.5:
            alpha = prob / (1 - prob)
        else:
            print("Caution! With the current alpha value ({}), you may trapped in local optima!".format(prob))
            print("It is suggested that probability value should be <= 0.5")
            alpha = prob / (1 - 0.49)
        self.max_alpha = alpha

        log_alpha = torch.log(torch.ones(dimension) * alpha)
        self.log_alpha = nn.Parameter(log_alpha)

        self.c = [1.16145124, -1.50204118, 0.58629921]

    def kl(self):
        alpha = torch.exp(self.log_alpha)

        kl = -(0.5 * self.log_alpha + self.c[0] * alpha + self.c[1] * alpha ** 2 + self.c[2] * alpha ** 3)

        return torch.mean(kl)

    def forward(self, x):
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
        kld = self.kl()

        if self.train():
            # Epsilon ~ N(0, 1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            alpha = torch.exp(self.log_alpha)

            # Epsilon ~ N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon, kld
        else:
            return x, kld

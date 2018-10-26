import torch.nn as nn
from dropout_models.gaussian_dropout import GaussianDropout
from dropout_models.variational_dropout import VariationalDropout


class Dropout(object):
    def __init__(self, keep_prob=0.5, dimension=None, dropout_type="bernoulli"):
        self.keep_prob = keep_prob
        self.dimension = dimension
        self.dropout_type = dropout_type
        self.dropout = self.create_dropout()

    def create_dropout(self):
        if self.dropout_type == "bernoulli":
            return nn.Dropout(self.keep_prob)
        elif self.dropout_type == "gaussian":
            return GaussianDropout(prob=self.keep_prob)
        elif self.dropout_type == "variational":
            return VariationalDropout(prob=self.keep_prob, dimension=self.dimension)

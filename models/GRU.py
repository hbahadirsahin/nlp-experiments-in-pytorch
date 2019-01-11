import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dropout_models.dropout import Dropout


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args_common = args["common_model_properties"]
        self.args_specific = args["gru"]

        self.hidden_dim = self.args_common["hidden_dim"]
        self.num_layers = self.args_common["num_layers"]
        self.batch_size = self.args_common["batch_size"]

        self.vocab = self.args_common["vocab"]

        # Device
        self.device = self.args_common["device"]

        # Input/Output dimensions
        self.embed_num = self.args_common["vocab_size"]
        self.embed_dim = self.args_common["embed_dim"]
        self.num_class = self.args_common["num_class"]

        # Embedding parameters
        self.padding_id = self.args_common["padding_id"]

        # Condition parameters
        self.use_pretrained_embed = self.args_common["use_pretrained_embed"]
        self.embed_train_type = self.args_common["embed_train_type"]
        self.bidirectional = self.args_specific["bidirectional"]
        self.rnn_bias = self.args_specific["bias"]

        # Pretrained embedding weights
        self.pretrained_weights = self.args_common["pretrained_weights"]

        # Dropout type
        self.dropout_type = self.args_specific["dropout_type"]

        # Dropout probabilities
        self.keep_prob = self.args_specific["keep_prob"]

        self.embed = self.initialize_embeddings()

        # It is NOT the inner GRU dropout!
        self.dropout = self.initialize_dropout()

        self.gru = nn.GRU(self.embed_dim,
                          self.hidden_dim,
                          dropout=self.keep_prob,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          bias=self.rnn_bias)

        self.hidden = self.init_hidden()

        if self.bidirectional is True:
            self.h2o = nn.Linear(self.hidden_dim * 2, self.num_class)
        else:
            self.h2o = nn.Linear(self.hidden_dim, self.num_class)

    def init_hidden(self):
        if self.bidirectional is True:
            return Variable(torch.zeros((1, self.batch_size, self.hidden_dim * 2)))
        else:
            return Variable(torch.zeros((1, self.batch_size, self.hidden_dim)))

    def initialize_embeddings(self):
        print("> Embeddings")
        embed = nn.Embedding(num_embeddings=self.embed_num,
                             embedding_dim=self.embed_dim,
                             padding_idx=self.padding_id).cpu()
        if self.use_pretrained_embed:
            print("> Pre-trained Embeddings")
            embed.from_pretrained(self.pretrained_weights)
        else:
            print("> Random Embeddings")
            random_embedding_weights = torch.rand(self.embed_num, self.embed_dim)
            embed.from_pretrained(random_embedding_weights)

        if self.embed_train_type == "static":
            print("> Static Embeddings")
            embed.weight.requires_grad = False
        elif self.embed_train_type == "nonstatic":
            print("> Non-Static Embeddings")
            embed.weight.requires_grad = True
        return embed

    def initialize_dropout(self):
        if self.dropout_type == "bernoulli" or self.dropout_type == "gaussian":
            print("> Dropout - ", self.dropout_type)
            return Dropout(keep_prob=self.keep_prob, dimension=None, dropout_type=self.dropout_type).dropout
        elif self.dropout_type == "variational":
            print("> Dropout - ", self.dropout_type)
            return Dropout(keep_prob=self.keep_prob, dimension=self.hidden_dim,
                           dropout_type=self.dropout_type).dropout
        else:
            print("> Dropout - Bernoulli (You provide undefined dropout type!)")
            return Dropout(keep_prob=self.keep_prob, dimension=None, dropout_type="bernoulli").dropout

    def forward(self, batch):
        kl_loss = torch.Tensor([0.0])

        x = self.embed(batch)
        x = self.dropout(x)
        x = x.view(len(x), self.batch_size, -1)

        if "cuda" in str(self.device):
            x = x.cuda()
        out, self.hidden = self.gru(x, self.hidden)
        out = torch.transpose(out, 0, 1)
        out = torch.transpose(out, 1, 2)

        out = F.max_pool1d(input=out, kernel_size=out.size(2)).squeeze(2)
        out = torch.tanh(out)

        out = self.h2o(out)
        out = F.log_softmax(out, dim=1)
        return out, kl_loss

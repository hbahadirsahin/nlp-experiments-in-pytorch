import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dropout_models.dropout import Dropout


class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args

        self.hidden_dim = args["rnn_hidden_dim"]
        self.num_layers = args["rnn_num_layers"]

        self.vocab = args["vocab"]

        # Device
        self.device = args["device"]

        # Input/Output dimensions
        self.embed_num = args["vocab_size"]
        self.embed_dim = args["embed_dim"]
        self.num_class = args["num_class"]

        # Embedding parameters
        self.padding_id = args["padding_id"]

        # Condition parameters
        self.use_pretrained_embed = args["use_pretrained_embed"]
        self.embed_train_type = args["embed_train_type"]
        self.bidirectional = args["rnn_bidirectional"]
        self.rnn_bias = args["rnn_bias"]
        self.use_batch_norm = args["use_batch_norm"]

        # Pretrained embedding weights
        self.pretrained_weights = args["pretrained_weights"]

        # Dropout type
        self.dropout_type = args["dropout_type"]

        # Dropout probabilities
        self.keep_prob = args["keep_prob"]

        # Batch normalization parameters
        batch_norm_momentum = args["batch_norm_momentum"]
        batch_norm_affine = args["batch_norm_affine"]

        self.embed = self.initialize_embeddings()

        # It is NOT the inner GRU dropout!
        self.dropout = self.initialize_dropout()

        self.gru = nn.GRU(self.embed_dim,
                          self.hidden_dim,
                          dropout=self.keep_prob,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional,
                          bias=self.rnn_bias)

        if self.use_batch_norm:
            print("> Batch Normalization")
            self.bn = nn.BatchNorm1d(self.hidden_dim, momentum=batch_norm_momentum, affine=batch_norm_affine)

        self.h2o = nn.Linear(self.hidden_dim, self.num_class)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros((1, batch_size, self.hidden_dim)))

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

        hidden = self.init_hidden(batch.size(1))

        x = self.embed(batch)
        x = x.view(len(x), x.size(1), -1)

        if "cuda" in str(self.device):
            x = x.cuda()
        out, hidden = self.gru(x, hidden)
        out = out.permute(1, 2, 0)

        out = F.max_pool1d(input=out, kernel_size=out.size(2)).squeeze(2)
        out = torch.tanh(out)

        logit = self.h2o(out)
        return logit, kl_loss

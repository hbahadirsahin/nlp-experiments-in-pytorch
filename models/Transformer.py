import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import clones


class LayerNormGoogle(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(LayerNormGoogle, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.epsilon) + self.b_2


class LayerNormOpenAI(nn.Module):
    def __init__(self, features, epsilon=1e-5):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std + self.epsilon) + self.b_2


class EncoderBlockGoogle(nn.Module):
    def __init__(self, layer, num_layers):
        super(EncoderBlockGoogle, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNormGoogle(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ResidualConnectionGoogle(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnectionGoogle, self).__init__()
        self.norm = LayerNormGoogle(size)
        # TODO: Use dropout interface
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, sublayer):
        return input + self.dropout(sublayer(self.norm(input)))


class EncoderLayerGoogle(nn.Module):
    def __init__(self, size, attention, feed_forward, dropout):
        self.size = size
        self.attention = attention
        self.feed_forward = feed_forward
        # Each encoder layer has two sublayers
        self.sublayer = clones(ResidualConnectionGoogle(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class EncoderClassifier(nn.Module):
    def __init__(self, embedding, encoder, classifier, is_average=True):
        super(EncoderClassifier, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.classifier = classifier
        self.is_average = is_average

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask)
        if self.is_average:
            # Averaged sentence representation
            x = torch.mean(x)
        x = self.classifier(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_model, d_hidden, num_classes, keep_prob):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class MultiHeadedAttentionGoogle(nn.Module):
    def __init__(self, heads=8, d_model=512, dropout=0.1):
        super(MultiHeadedAttentionGoogle, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        # Dot product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [linear(x).view(num_batches, -1, self.heads, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.heads * self.d_k)

        return self.linears[-1](x)


class PositionalFeedForwardGoogle(nn.Module):
    def __init__(self, d_model, d_ff, keep_prob=0.1):
        super(PositionalFeedForwardGoogle, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(keep_prob)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.w_2(self.dropout(self.relu(self.w_1(input))))


class Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, padding_id, use_pretrained_embed, pretrained_weights):
        super(Embeddings, self).__init__()
        # Initialize embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_id).cpu()
        if use_pretrained_embed:
            self.embedding.from_pretrained(pretrained_weights)
        self.embed_dim = embed_dim

    def forward(self, input):
        return self.embedding(input)


class PositionalEncodingGoogle(nn.Module):
    def __init__(self, d_model, keep_prob=0.1, max_len=5000):
        super(PositionalEncodingGoogle, self).__init__()
        self.dropout = nn.Dropout(keep_prob)

        positional_encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0., max_len).unsqueeze(1)
        # Log space
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-math.log(10000) / d_model))

        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("pe", positional_encoding)

    def forward(self, input):
        return self.dropout(input + Variable(self.pe[:, :input.size(1)], requires_grad=False))


class TransformerGoogle(nn.Module):
    def __init__(self, args):
        super(TransformerGoogle, self).__init__()

        self.args = args

        # Input/Output dimensions
        self.vocab_size = args["vocab_size"]
        self.embed_dim = args["embed_dim"]
        self.num_class = args["num_class"]

        # Embedding parameters
        self.padding_id = args["padding_id"]

        # Condition parameters
        self.use_pretrained_embed = args["use_pretrained_embed"]

        # Pretrained embedding weights
        self.pretrained_weights = args["pretrained_weights"]

        # Dropout probabilities
        self.keep_prob = args["keep_prob"]

        self.transformer_type = args["transformer_type"]

        if self.transformer_type == "classifier":
            self.create_classifier_transformer()

    def create_classifier_transformer(self, heads=8, num_encoder_layers=6, d_ff=2048):
        c = copy.deepcopy()

        # Initialize blocks for the full model
        attention = MultiHeadedAttentionGoogle(h=heads, d_model=self.embed_dim)
        ff = PositionalFeedForwardGoogle(d_model=self.embed_dim, d_ff=d_ff)
        embeddings = Embeddings(self.embed_dim, self.vocab_size, self.padding_id, self.use_pretrained_embed,
                                self.pretrained_weights)
        positional_embeddings = PositionalEncodingGoogle(d_model=self.embed_dim)

        # Initialize full model
        model = EncoderClassifier(nn.Sequential(embeddings, c(positional_embeddings)),
                                  EncoderBlockGoogle(
                                      EncoderLayerGoogle(self.embed_dim, c(attention), c(ff), self.keep_prob),
                                      num_encoder_layers),
                                  Classifier(self.embed_dim, d_hidden=self.embed_dim // 2, num_classes=self.num_class))

        # Initialize model parameters
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def forward(self, x):
        return x


if __name__ == '__main__':
    print("Transformer tests")
    plt.figure(figsize=(15, 5))
    pe = PositionalEncodingGoogle(20, 0)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

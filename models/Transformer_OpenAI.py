import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import clones



class Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, keep_prob, padding_id, use_pretrained_embed, pretrained_weights):
        super(Embeddings, self).__init__()
        # Initialize embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_id).cpu()
        if use_pretrained_embed:
            self.load_pretrained_weights()
        self.embed_drop = nn.Dropout(keep_prob)

    def forward(self, input):
        x = self.embed_drop(self.embedding(input))
        out = x.sum(dim=2)
        return out


class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-5):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std + self.epsilon) + self.b_2


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_state, embed_dim, keep_prob):
        self.fc = nn.Conv1d(num_state, 1, embed_dim)
        self.proj = nn.Conv1d(embed_dim, 1, num_state)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob)

    def forward(self, input):
        x = self.activation(self.fc(input))
        x = self.dropout(self.proj(x))
        return x


class ModifiedMultiHeadedAttention(nn.Module):
    def __init__(self, num_state, n_ctx, num_heads, keep_prob_attention, keep_prob_residual, scale=False):
        assert num_state % num_heads == 0
        self.bias = torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx)
        self.num_heads = num_heads
        self.split_size = num_state
        self.scale = scale
        self.attn = nn.Conv1d(num_state * 3, 1, num_state)
        self.proj = nn.Conv1d(num_state, 1, num_state)
        self.attn_dropout = nn.Dropout(keep_prob_attention)
        self.residual_dropout = nn.Dropout(keep_prob_residual)

    def attention(self, query, key, value):
        weight = torch.matmul(query, key)
        if self.scale:
            weight = weight / math.sqrt(value.size(-1))

        # Mask attention weights
        bias = self.bias[:, :, :weight.size(-2), :weight.size(-1)]
        weight = weight * bias - 1e9 * (1 - bias)

        p_attn = F.softmax(weight, dim=-1)
        if self.attn_dropout is not None:
            p_attn = self.attn_dropout(p_attn)
        return torch.matmul(p_attn, value)

    # Direct c/p from huggingface, which is the equivalent of original tensorflow implementation.
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    # Direct c/p from huggingface, which is the equivalent of original tensorflow implementation.
    def split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if is_key:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, input):
        x = self.attn(input)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        out = self.proj(self.merge_heads(self.attention(query, key, value)))
        return self.residual_dropout(out)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, keep_prob_attention, keep_prob_residual, keep_prob_mlp, n_ctx=512,
                 scale=False, use_builtin_mha=False):
        if use_builtin_mha:
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                   num_heads=num_heads,
                                                   dropout=keep_prob_attention)
        else:
            self.attention = ModifiedMultiHeadedAttention(num_state=embed_dim,
                                                          n_ctx=n_ctx,
                                                          num_heads=num_heads,
                                                          keep_prob_attention=keep_prob_attention,
                                                          keep_prob_residual=keep_prob_residual,
                                                          scale=scale)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.mlp = MultiLayerPerceptron(4 * embed_dim, embed_dim, keep_prob_mlp)
        self.layer_norm2 = LayerNorm(embed_dim)

    def forward(self, input):
        x = self.attn(input)
        x_hat = self.ln_1(input + x)
        x = self.mlp(x_hat)
        x = self.ln_2(x_hat + x)
        return x

class LanguageModelHead(nn.Module):
    def __init__(self, embedding, embed_dim):
        super(LanguageModelHead, self).__init__()
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(in_features=embedding.embedding.weight.shape[1],
                                out_features=embedding.embedding.weight.shape[0],
                                bias=True)
        self.decoder.weight = embedding.embedding.weight

    def forward(self, input):
        # Remove last token
        x = input[:, :-1].view(-1, self.embed_dim)
        x = self.decoder(x)
        return x


class TransformerOpenAI:
    def __init__(self, args):
        super(TransformerOpenAI, self).__init__()

        self.args_common = args["common_model_properties"]
        self.args_specific = args["transformer_openai"]

        # Device
        self.device = self.args_common["device"]

        # Input/Output dimensions
        self.vocab_size = self.args_common["vocab_size"]
        self.embed_dim = self.args_common["embed_dim"]
        self.num_class = self.args_common["num_class"]

        # Embedding parameters
        self.padding_id = self.args_common["padding_id"]

        # Condition parameters
        self.use_pretrained_embed = self.args_common["use_pretrained_embed"]

        # Model/Context size
        self.d_model = self.args_specific["d_model"]

        # Dropout probabilities for each individual part of the full model.
        self.keep_prob_embed = self.args_specific["keep_prob_embed"]

        # Number of parallel attention layers for MultiHeadedAttention
        self.heads = self.args_specific["heads"]

        # Number of layers in terms of Blocks
        self.num_layers = self.args_specific["num_layers"]

        if self.transformer_type == "classifier":
            self.model = self.create_classifier_transformer()
        else:
            raise ValueError("Transformer can be created as classifier for now!")

    def create_classifier_transformer(self):
        c = copy.deepcopy

        embedding = Embeddings(self.embed_dim, self.vocab_size, self.keep_prob_embed, self.padding_id,
                               self.use_pretrained_embed, self.pretrained_weights)



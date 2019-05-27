import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import clones


class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-5):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std + self.epsilon) + self.b_2


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


class LanguageModelHead(nn.Module):
    def __init__(self):
        super(LanguageModelHead, self).__init__()


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



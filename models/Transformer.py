import torch
import torch.nn as nn

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

    def forward(self, input, mask):
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


class DecoderBlockGoogle(nn.Module):
    def __init__(self, layer, num_layers):
        super(DecoderBlockGoogle, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNormGoogle(layer.size)

    def forward(self, input, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(input, memory, source_mask, target_mask)
        return self.norm(x)


class DecoderLayerGoogle(nn.Module):
    def __init__(self, size, attention, source_attention, feed_forward, dropout):
        super(DecoderLayerGoogle, self).__init__()
        self.size = size
        self.attention = attention
        self.source_attention = source_attention
        self.feed_forward = feed_forward
        # Each decoder layer has three sublayers
        self.sublayer = clones(ResidualConnectionGoogle(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.attention(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.attention(x, memory, memory, source_mask))
        return self.sublayer[2](x, self.feed_forward)


class TransformerGoogle(nn.Module):
    def __init__(self, args):
        super(TransformerGoogle, self).__init__()

        self.args = args


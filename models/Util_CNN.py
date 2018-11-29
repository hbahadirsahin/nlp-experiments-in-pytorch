import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channel_size, filter_count, filter_size, stride):
        super(ConvolutionalBlock, self).__init__()
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(num_features=filter_count)
        conv1 = nn.Conv1d(in_channels=input_channel_size,
                          out_channels=filter_count,
                          kernel_size=filter_size,
                          stride=stride,
                          padding=1)
        conv2 = nn.Conv1d(in_channels=filter_count,
                          out_channels=filter_count,
                          kernel_size=filter_size,
                          stride=1,
                          padding=1)
        self.block = nn.Sequential(conv1, bn, relu, conv2, bn, relu)

    def forward(self, input):
        return self.block(input)


class KMaxPooling(nn.Module):
    def __init__(self, k):
        super(KMaxPooling, self).__init__()
        assert 1 < k
        self.k = k

    def forward(self, input):
        kmax, _ = input.topk(input.shape(2) // self.k, dim=2)
        return kmax


class LayerBlock(nn.Module):
    def __init__(self, input_channel_size, filter_count, conv_filter_size, maxpool_filter_size, kmax_k=2,
                 downsample=False, downsample_type="resnet", use_shortcut=True):
        super(LayerBlock, self).__init__()
        self.downsample = downsample
        self.use_shortcut = use_shortcut

        self.pool = None
        stride = 1
        if self.downsample:
            if downsample_type == "resnet":
                stride = 2
            elif downsample_type == "vgg":
                self.pool = nn.MaxPool1d(kernel_size=maxpool_filter_size, stride=2, padding=1)
            elif downsample_type == "kmax":
                self.pool = self.KMaxPooling(k=kmax_k)
            else:
                raise KeyError("Downsample_type can be (1) resnet, (2) vgg, or (3) kmax")

        self.convolutional_block = self.ConvolutionalBlock(input_channel_size=input_channel_size,
                                                           filter_count=filter_count,
                                                           filter_size=conv_filter_size,
                                                           stride=stride)

        if use_shortcut and self.downsample:
            self.shortcut = nn.Conv1d(in_channels=input_channel_size,
                                      out_channels=filter_count,
                                      kernel_size=1,
                                      stride=2)

    def forward(self, input):
        residual = input
        if self.downsample and self.pool:
            x = self.pool(input)
        x = self.convolutional_block(x)

        if self.downsample and self.use_shortcut:
            residual = self.shortcut(residual)

        if self.use_shortcut:
            x += residual
        return x

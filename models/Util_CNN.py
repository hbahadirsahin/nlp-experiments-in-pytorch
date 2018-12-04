import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ConvolutionEncoder(nn.Module):
    def __init__(self, args):
        super(ConvolutionEncoder, self).__init__()
        self.args = args

        # Device
        self.device = args["device"]

        # Input/Output dimensions
        self.vocab_size = args["vocab_size"]
        self.embed_dim = args["embed_dim"]

        # Embedding parameters
        self.padding_id = args["padding_id"]

        # Condition parameters
        self.use_pretrained_embed = args["use_pretrained_embed"]
        self.use_batch_norm = args["use_batch_norm"]

        # Pretrained embedding weights
        self.pretrained_weights = args["pretrained_weights"]

        # Batch normalization parameters
        self.batch_norm_momentum = args["batch_norm_momentum"]
        self.batch_norm_affine = args["batch_norm_affine"]

        # Convolution parameters
        self.input_channel = 1
        self.filter_counts = args["encodercnn_filter_counts"]
        self.filter_sizes = args["encodercnn_filter_sizes"]
        self.strides = args["encodercnn_strides"]

        # Initialize embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_id).cpu()
        if self.use_pretrained_embed:
            print("> Pre-trained Embeddings")
            self.embedding.from_pretrained(self.pretrained_weights)

        # Initialize convolutions
        self.conv1 = nn.Conv2d(in_channels=self.input_channel,
                               out_channels=self.filter_counts[0],
                               kernel_size=(self.filter_sizes[0], self.embed_dim),
                               stride=self.strides[0],
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.filter_counts[0],
                               out_channels=self.filter_counts[1],
                               kernel_size=(self.filter_sizes[1], 1),
                               stride=self.strides[1],
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=self.filter_counts[1],
                               out_channels=self.filter_counts[2],
                               kernel_size=(self.filter_sizes[2], 1),
                               stride=self.strides[2],
                               bias=True)

        # Initialize batch norms
        if self.use_batch_norm:
            self.conv1_bn = nn.BatchNorm2d(num_features=self.filter_counts[0],
                                           momentum=self.batch_norm_momentum,
                                           affine=self.batch_norm_affine)
            self.conv2_bn = nn.BatchNorm2d(num_features=self.filter_counts[1],
                                           momentum=self.batch_norm_momentum,
                                           affine=self.batch_norm_affine)

        # Well, self-explanatory.
        self.relu = nn.ReLU()

    def forward(self, batch):
        batch_permuted = batch.permute(1, 0)
        h = self.embed(batch_permuted)
        if "cuda" in str(self.device):
            h = h.cuda()

        if self.use_batch_norm:
            h = self.relu(self.conv1_bn(self.conv1(h)))
            h = self.relu(self.conv2_bn(self.conv2(h)))
            h = self.relu(self.conv3(h))
        else:
            h = self.relu(self.conv1(h))
            h = self.relu(self.conv2(h))
            h = self.relu(self.conv3(h))

        return h, self.embed


class DeconvolutionDecoder(nn.Module):
    def __init__(self, args, embedding):
        super(DeconvolutionDecoder, self).__init__()
        self.args = args

        # Device
        self.device = args["device"]

        # Input/Output dimensions
        self.embed_dim = args["embed_dim"]

        # Embedding initialized in Encoder
        self.embedding = embedding

        # Condition parameters
        self.use_batch_norm = args["use_batch_norm"]

        # Batch normalization parameters
        self.batch_norm_momentum = args["batch_norm_momentum"]
        self.batch_norm_affine = args["batch_norm_affine"]

        # Convolution parameters
        self.input_channel = 1
        self.filter_counts = list(reversed(args["encodercnn_filter_counts"]))
        self.filter_sizes = list(reversed(args["encodercnn_filter_sizes"]))
        self.strides = list(reversed(args["encodercnn_strides"]))
        self.temperature = args["deconv_temperature"]

        # Initialize deconvolutions
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.filter_counts[0],
                                          out_channels=self.filter_counts[1],
                                          kernel_size=(self.filter_sizes[0], 1),
                                          stride=self.strides[0],
                                          bias=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.filter_counts[1],
                                          out_channels=self.filter_counts[2],
                                          kernel_size=(self.filter_sizes[1], 1),
                                          stride=self.strides[1],
                                          bias=True)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.filter_counts[2],
                                          out_channels=self.input_channel,
                                          kernel_size=(self.filter_sizes[2], self.embed_dim),
                                          stride=self.strides[2],
                                          bias=True)

        # Initialize batch norms
        if self.use_batch_norm:
            self.deconv1_bn = nn.BatchNorm2d(num_features=self.filter_counts[0],
                                             momentum=self.batch_norm_momentum,
                                             affine=self.batch_norm_affine)
            self.deconv2_bn = nn.BatchNorm2d(num_features=self.filter_counts[1],
                                             momentum=self.batch_norm_momentum,
                                             affine=self.batch_norm_affine)

        # Well, self-explanatory.
        self.relu = nn.ReLU()

    def forward(self, h):
        if self.use_batch_norm:
            x_ = self.relu(self.deconv1_bn(self.deconv1(h)))
            x_ = self.relu(self.deconv2_bn(self.deconv2(x_)))
            x_ = self.relu(self.deconv3(x_))
        else:
            x_ = self.relu(self.deconv1(h))
            x_ = self.relu(self.deconv2(x_))
            x_ = self.relu(self.deconv3(x_))

        x_ = x_.squeeze()

        # p(w^t = v): Probability of w^t to be word v, as w^t is the t'th word of the reconstructed sentence.
        normalized_x_ = torch.norm(x_, p=2, dim=2, keepdim=True)
        reconstructed_x_ = x_ / normalized_x_

        normalized_w = (nn.Variable(self.embedding.weight.data).t()).unsqueeze(0)
        normalized_w = normalized_w.expand(reconstructed_x_.size(0), *normalized_w.size())
        probs = torch.bmm(reconstructed_x_, normalized_w) / self.temperature
        # Reconstruction log probabilities (not loss)
        return F.log_softmax(probs, dim=2)


class FullyConnectedNetwork(nn.Module):
    def __init__(self, args, input_size):
        super(FullyConnectedNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = args["conv_deconv_hidden_layer_size"]
        self.num_class = args["num_class"]
        self.keep_prob = args["conv_deconv_keep_prob"]

        self.fc1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.num_class)

        self.dropout = nn.Dropout(self.keep_prob)

    def forward(self, input):
        x = self.dropout(self.fc1(input))
        x = self.fc2(x)
        # Supervised log probabilities
        return F.log_softmax(x, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

from Util_CNN import KMaxPooling, LayerBlock
from dropout_models.dropout import Dropout


class TextCnn(nn.Module):
    def __init__(self, args):
        super(TextCnn, self).__init__()
        self.args = args

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
        self.use_padded_conv = args["use_padded_conv"]
        self.use_batch_norm = args["use_batch_norm"]

        # Pretrained embedding weights
        self.pretrained_weights = args["pretrained_weights"]

        # Dropout type
        self.dropout_type = args["dropout_type"]

        # Dropout probabilities
        keep_prob = args["keep_prob"]

        # Batch normalization parameters
        self.batch_norm_momentum = args["batch_norm_momentum"]
        self.batch_norm_affine = args["batch_norm_affine"]

        # Convolution parameters
        self.input_channel = 1
        self.filter_count = args["filter_count"]
        self.filter_sizes = args["filter_sizes"]

        # Embedding Layer Initialization
        if self.embed_train_type == "multichannel":
            self.embed, self.embed_static = self.initialize_embeddings()
        else:
            self.embed, _ = self.initialize_embeddings()

        # Convolution Initialization
        self.convs = self.initialize_conv_layer()

        # Initialize convolution weights
        self.initialize_weights()

        # Flatten conv layers' output
        num_flatten_feature = len(self.filter_sizes) * self.filter_count

        # Batch Normalization initialization
        if self.use_batch_norm:
            print("> Batch Normalization")
            self.initialize_batch_normalization(num_flatten_feature)

        # Dropout initialization
        if self.dropout_type == "bernoulli" or self.dropout_type == "gaussian":
            print("> Dropout - ", self.dropout_type)
            self.dropout = Dropout(keep_prob=keep_prob, dimension=None, dropout_type=self.dropout_type).dropout
        elif self.dropout_type == "variational":
            print("> Dropout - ", self.dropout_type)
            self.dropout_before_flatten = Dropout(keep_prob=0.2, dimension=num_flatten_feature,
                                                  dropout_type=self.dropout_type).dropout
            self.dropout_fc1 = Dropout(keep_prob=keep_prob, dimension=num_flatten_feature // 2,
                                       dropout_type=self.dropout_type).dropout
        else:
            print("> Dropout - Bernoulli (You provide undefined dropout type!)")
            self.dropout = Dropout(keep_prob=keep_prob, dimension=None, dropout_type="bernoulli").dropout

        # Fully Connected Layer 1 initialization
        self.fc1 = nn.Linear(in_features=num_flatten_feature,
                             out_features=num_flatten_feature // 2,
                             bias=True)

        # Fully Connected Layer 2 initialization
        self.fc2 = nn.Linear(in_features=num_flatten_feature // 2,
                             out_features=self.num_class,
                             bias=True)

    def initialize_embeddings(self):
        print("> Embeddings")
        embed = nn.Embedding(num_embeddings=self.embed_num,
                             embedding_dim=self.embed_dim,
                             padding_idx=self.padding_id).cpu()

        embed_static = None
        # Create 2nd embedding layer for multichannel purpose
        if self.embed_train_type == "multichannel":
            embed_static = nn.Embedding(num_embeddings=self.embed_num,
                                        embedding_dim=self.embed_dim,
                                        padding_idx=self.padding_id).cpu()

        if self.use_pretrained_embed:
            print("> Pre-trained Embeddings")
            embed.from_pretrained(self.pretrained_weights)
            if self.embed_train_type == "multichannel":
                embed_static.from_pretrained(self.pretrained_weights)
        else:
            print("> Random Embeddings")
            random_embedding_weights = torch.rand(self.embed_num, self.embed_dim)
            embed.from_pretrained(random_embedding_weights)
            if self.embed_train_type == "multichannel":
                embed_static.from_pretrained(random_embedding_weights)

        if self.embed_train_type == "static":
            print("> Static Embeddings")
            embed.weight.requires_grad = False
        elif self.embed_train_type == "nonstatic":
            print("> Non-Static Embeddings")
            embed.weight.requires_grad = True
        elif self.embed_train_type == "multichannel":
            embed.weight.requires_grad = True
            embed_static.weight.requires_grad = False
        else:
            raise KeyError("Embedding train type can be (1) static, (2) nonstatic or (3) multichannel")
        return embed, embed_static

    def initialize_conv_layer(self):
        if self.use_padded_conv:
            print("> Padded convolution")
            return nn.ModuleList([nn.Conv2d(in_channels=self.input_channel,
                                            out_channels=self.filter_count,
                                            kernel_size=(filter_size, self.embed_dim),
                                            stride=(1, 1),
                                            padding=(filter_size // 2, 0),
                                            bias=True) for filter_size in self.filter_sizes])
        else:
            print("> Without-pad convolution")
            return nn.ModuleList([nn.Conv2d(in_channels=self.input_channel,
                                            out_channels=self.filter_count,
                                            kernel_size=(filter_size, self.embed_dim),
                                            bias=True) for filter_size in self.filter_sizes])

    def initialize_weights(self):
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            conv.bias.data.fill_(0.01)

    def initialize_batch_normalization(self, num_flatten_feature):
        self.convs_bn = nn.BatchNorm2d(num_features=self.filter_count,
                                       momentum=self.batch_norm_momentum,
                                       affine=self.batch_norm_affine)
        self.fc1_bn = nn.BatchNorm1d(num_features=num_flatten_feature // 2,
                                     momentum=self.batch_norm_momentum,
                                     affine=self.batch_norm_affine)
        self.fc2_bn = nn.BatchNorm1d(num_features=self.num_class,
                                     momentum=self.batch_norm_momentum,
                                     affine=self.batch_norm_affine)

    def forward(self, batch):
        kl_loss = torch.Tensor([0.0])
        # Input shape: [sentence_length, batch_size]
        batch_permuted = batch.permute(1, 0)
        # X shape: [batch_size, sentence_length]
        x = self.embed(batch_permuted)
        if self.embed_train_type == "multichannel":
            x_static = self.embed_static(batch_permuted)
            x = torch.stack[(x_static, x), 1]
        if "cuda" in str(self.device):
            x = x.cuda()
            kl_loss = kl_loss.cuda()
        # X shape: [batch_size, sentence_length, embedding_dim]
        x = x.unsqueeze(1)
        # X shape: [batch_size, 1, sentence_length, embedding_dim]
        if self.use_batch_norm:
            x = [self.convs_bn(F.relu(conv(x))).squeeze(3) for conv in self.convs]
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # X[i] shape: [batch_size, filter_count, sentence_length - filter_size[i]]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        # X[i] shape: [batch_size, filter_count]
        if self.dropout_type == "variational":
            x, kld = self.dropout_before_flatten(torch.cat(x, dim=1))
            kl_loss += kld.sum()
        else:
            x = self.dropout(torch.cat(x, dim=1))
        # Fully Connected Layers
        if self.use_batch_norm:
            if self.dropout_type == "variational":
                x, kld = self.dropout_fc1(self.fc1_bn(F.relu(self.fc1(x))))
                kl_loss += kld.sum()
            else:
                x = self.dropout(self.fc1_bn(F.relu(self.fc1(x))))
            x = self.fc2_bn(self.fc2(x))
        else:
            if self.dropout_type == "variational":
                x, kld = self.dropout_fc1(F.relu(self.fc1(x)))
                kl_loss += kld.sum()
            else:
                x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
        return x, kl_loss


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()

        self.args = args

        # Device
        self.device = args["device"]

        # Input/Output dimensions
        self.vocab_size = args["vocab_size"]
        self.embed_dim = args["embed_dim"]
        self.num_class = args["num_class"]

        # Embedding parameters
        self.padding_id = args["padding_id"]

        # Dropout type
        self.dropout_type = args["dropout_type"]

        # Dropout probabilities
        self.keep_prob = args["keep_prob"]

        # CharCNN specific parameters
        self.max_sequence_length = args["max_sequence_length"]

        if args["feature_size"] == "large":
            self.filter_count = 1024
            self.linear_unit_count = 2048
        elif args["feature_size"] == "small":
            self.filter_count = 256
            self.linear_unit_count = 1024
        else:
            self.filter_count = args["filter_count"]
            self.linear_unit_count = args["linear_unit_count"]

        self.filter_sizes = args["filter_sizes"]
        self.max_pool_kernels = args["max_pool_kernels"]

        # Embedding initialization
        # As the original CharCNN paper, I initialized char embeddings as one-hot vector.
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_id)
        self.embedding.weight.data = torch.eye(self.vocab_size, self.embed_dim)
        self.embedding.weight.reqiures_grad = False

        # Convolution Layer 1
        self.conv1 = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[0])
        self.pool1 = nn.MaxPool1d(kernel_size=self.max_pool_kernels[0])

        # Convolution Layer 2
        self.conv2 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[1])
        self.pool2 = nn.MaxPool1d(kernel_size=self.max_pool_kernels[1])

        # Convolution Layer 3
        self.conv3 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[2])

        # Convolution Layer 4
        self.conv4 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[3])

        # Convolution Layer 5
        self.conv5 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[4])

        # Convolution Layer 6
        self.conv6 = nn.Conv1d(in_channels=self.filter_count, out_channels=self.filter_count,
                               kernel_size=self.filter_sizes[5])
        self.pool3 = nn.MaxPool1d(kernel_size=self.max_pool_kernels[2])

        # Activation
        self.relu = nn.ReLU()

        # Number of features after convolution blocks
        num_features = (self.max_sequence_length - 96) // 27 * self.filter_count

        self.initialize_dropout(num_features)

        # Linear Block 1
        self.linear1 = nn.Linear(num_features, self.linear_unit_count)

        # Linear Block 2
        self.linear2 = nn.Linear(self.linear_unit_count, self.linear_unit_count)

        # Linear Block 3
        self.linear3 = nn.Linear(self.linear_unit_count, self.num_class)

    def initialize_dropout(self, num_features):
        # Dropout initialization
        if self.dropout_type == "bernoulli" or self.dropout_type == "gaussian":
            print("> Dropout - ", self.dropout_type)
            self.dropout = Dropout(keep_prob=self.keep_prob, dimension=None, dropout_type=self.dropout_type).dropout
        elif self.dropout_type == "variational":
            print("> Dropout - ", self.dropout_type)
            self.dropout = Dropout(keep_prob=self.keep_prob, dimension=num_features,
                                   dropout_type=self.dropout_type).dropout
        else:
            print("> Dropout - Bernoulli (You provide undefined dropout type!)")
            self.dropout = Dropout(keep_prob=self.keep_prob, dimension=None, dropout_type="bernoulli").dropout

    def forward(self, batch):
        kl_loss = torch.Tensor([0.0])
        # Get batch size to beginning
        x = batch.permute(1, 0)
        # Embedding magic
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        if "cuda" in str(self.device):
            kl_loss = kl_loss.cuda()
        # To Convolution-Pooling
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(self.relu(self.conv6(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        # To Linear
        if self.dropout_type == "variational":
            x, kld = self.dropout(self.relu(self.linear1(x)))
            kl_loss += kld.sum()
        else:
            x = self.dropout(self.relu(self.linear1(x)))
        if self.dropout_type == "variational":
            x, kld = self.dropout(self.relu(self.linear2(x)))
            kl_loss += kld.sum()
        else:
            x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)

        return x, kl_loss


class VDCNN(nn.Module):
    def __init__(self, args):
        super(VDCNN, self).__init__()

        self.args = args

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
        self.use_shortcut = args["use_shortcut"]

        # Pretrained embedding weights
        self.pretrained_weights = args["pretrained_weights"]

        # Dropout type
        self.dropout_type = args["dropout_type"]

        # Dropout probabilities
        self.keep_prob = args["keep_prob"]

        # Batch normalization parameters
        self.batch_norm_momentum = args["batch_norm_momentum"]
        self.batch_norm_affine = args["batch_norm_affine"]

        # Convolution parameters
        self.depth = args["depth"]
        assert self.depth in [9, 17, 29, 49]
        self.filter_counts = args["filter_count"]
        self.filter_size = args["filter_size"]

        # Downsampling parameters
        self.downsampling_type = args["downsampling_type"]
        self.maxpool_filter_size = args["maxpool_filter_size"]
        self.k = args["kmax"]

        number_of_layers = self.initialize_number_of_layers()
        layers = nn.ModuleList()

        first_conv_layer = nn.Conv1d(in_channels=self.embed_dim,
                                     out_channels=self.filter_count[0],
                                     kernel_size=self.filter_size,
                                     padding=1)
        layers.append(first_conv_layer)

        # Add second convolution layer block where input_size is self.filter_count[0], output_size is self.filter_count[0]
        for n in range(number_of_layers[0]):
            layers.append(LayerBlock(input_channel_size=self.filter_count[0],
                                     filter_count=self.filter_count[0],
                                     conv_filter_size=self.filter_size,
                                     maxpool_filter_size=self.maxpool_filter_size,
                                     kmax_k=self.k,
                                     downsample_type=self.downsampling_type,
                                     use_shortcut=self.use_shortcut))

        # Add third convolution layer block where input_size is self.filter_count[0], output_size is self.filter_count[1]
        layers.append(LayerBlock(input_channel_size=self.filter_count[0],
                                 filter_count=self.filter_count[1],
                                 conv_filter_size=self.filter_size,
                                 maxpool_filter_size=self.maxpool_filter_size,
                                 kmax_k=self.k,
                                 downsample_type=self.downsampling_type,
                                 downsample=True,
                                 use_shortcut=self.use_shortcut))
        for n in range(number_of_layers[1] - 1):
            layers.append(LayerBlock(input_channel_size=self.filter_count[1],
                                     filter_count=self.filter_count[1],
                                     conv_filter_size=self.filter_size,
                                     maxpool_filter_size=self.maxpool_filter_size,
                                     kmax_k=self.k,
                                     downsample_type=self.downsampling_type,
                                     use_shortcut=self.use_shortcut))

        # Add fourth convolution layer block where input_size is self.filter_count[1], output_size is self.filter_count[2]
        layers.append(LayerBlock(input_channel_size=self.filter_count[1],
                                 filter_count=self.filter_count[2],
                                 conv_filter_size=self.filter_size,
                                 maxpool_filter_size=self.maxpool_filter_size,
                                 kmax_k=self.k,
                                 downsample_type=self.downsampling_type,
                                 downsample=True,
                                 use_shortcut=self.use_shortcut))
        for n in range(number_of_layers[2] - 1):
            layers.append(LayerBlock(input_channel_size=self.filter_count[2],
                                     filter_count=self.filter_count[2],
                                     conv_filter_size=self.filter_size,
                                     maxpool_filter_size=self.maxpool_filter_size,
                                     kmax_k=self.k,
                                     downsample_type=self.downsampling_type,
                                     use_shortcut=self.use_shortcut))

        # Add fifth convolution layer block where input_size is self.filter_count[2], output_size is self.filter_count[3]
        layers.append(LayerBlock(input_channel_size=self.filter_count[2],
                                 filter_count=self.filter_count[3],
                                 conv_filter_size=self.filter_size,
                                 maxpool_filter_size=self.maxpool_filter_size,
                                 kmax_k=self.k,
                                 downsample_type=self.downsampling_type,
                                 downsample=True,
                                 use_shortcut=self.use_shortcut))
        for n in range(number_of_layers[2] - 1):
            layers.append(LayerBlock(input_channel_size=self.filter_count[3],
                                     filter_count=self.filter_count[3],
                                     conv_filter_size=self.filter_size,
                                     maxpool_filter_size=self.maxpool_filter_size,
                                     kmax_k=self.k,
                                     downsample_type=self.downsampling_type,
                                     use_shortcut=self.use_shortcut))

        self.all_conv_layers == nn.Sequential(*layers)
        self.kmax_pooling == KMaxPooling(k=self.k)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.filter_counts[3] * self.k, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, self.num_class)

    def initialize_number_of_layers(self):
        if self.depth == 9:
            return [2] * 4
        elif self.depth == 17:
            return [4] * 4
        elif self.depth == 29:
            return [10, 10, 4, 4]
        elif self.depth == 49:
            return [16, 16, 10, 6]

    def forward(self, batch):
        kl_loss = torch.Tensor([0.0])
        # Get batch size to beginning
        x = batch.permute(1, 0)
        # Embedding magic
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        if "cuda" in str(self.device):
            kl_loss = kl_loss.cuda()
        x = self.all_conv_layers(x)
        x = self.kmax_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x, kl_loss

# README

This folder only holds a configuration JSON file. This file is used to define all properties and parameters that are
needed to be execute this project.

## Configuration Details

`config.json` has 4 main property dictionaries:

- `dataset_properties` defines the dataset reading/loading/preparing related properties.
- `training_properties` defines all training-related properties and parameters algorithm/optimizer type, learning rate,
decay rate, and so on.
- `evaluation_properties` defines the necessary model/vocabulary paths for evaluation purposes.
- `model_properties` defines anything that is needed to create a model.

### Dataset Properties

There are 10 properties under this `dataset_properties`:

- `data_path`: The original dataset file path (The current version of the code splits a full dataset to train/validation/test sets. But does not allow you to use an already splitted version).
- `stop_word_path`: The stop word file path.
- `embedding_vector`: Embedding alias that torchtext needs/uses while building vocabulary (predefined aliases can be found torchtext's vocab.py file). For instance "fasttext.en.300d", "fasttext.tr.300d", etc.
- `vector_cache`: The embedding file path that torchtext creates. To prevent it to download same file over and over again, you need to provide its path.
- `pretrained_embedding_path`: This is the original, Gensim readable, embedding files' path (note that only use case for this is Fasttext-based OOV word generation).
- `checkpoint_path`: The path for saved model file that you want to continue your training.
- `oov_embedding_type`: The property to define how to handle out-of-vocabulary word embeddings. It takes a string and can be "zeros", "ones", "random", "uniform" or "fasttext_oov".
- `batch_size`: Self-explanatory.
- `fixed_length`: Integer parameter to define the maximum length of an input string (lets say for a sentence, it limits the number of words). For values lower than or equal to 0, the dataset loader uses dynamic input size. It is for reducing the memory consumption.
- `min_freq`: Integer parameter to define the minimum frequency condition on words to be appear in vocabulary. For values lower than or equal to 0, the dataset loader will assign every seen word (min_freq=1) in vocabulary. It is for reducing the memory consumption.

### Training Properties

There are 14 properties under this `training_properties` which determines the learning algorithm, optimizer, optimizer's
parameters and printing/saving/evaluating related stuff:

- `learner`: String parameter to choose which learning algorithm to use. It can be "text_cnn", "gru", "lstm", "char_cnn",
"vdcnn", "conv_deconv_cnn" and "transformer_google" (Last Update: 15-01-2019) 
- `task`: String parameter to choose which task the learner will be trained on. It can be "classification" or "ner".
- `optimizer`: String parameter to choose which optimizer to use. It can be "Adam", "SGD", "OpenAIAdam", "Noam", and "Padam".
- `learning_rate`: Self-explanatory. Takes float value.
- `scheduler_type`: String parameter to choose a scheduler for OpenAIAdam optimizer (it has no usage for others). It can
be "cos", "constant" or "linear".
- `amsgrad`: Boolean parameter to choose whether to use amsgrad or not in Padam optimizer.
- `partial_adam`: Float parameter to define the "partial" parameter in Padam. It can take values between (0, 0.5]
- `weight_decay`: Float parameter for L2 normalization term. *Note that for my test cases, any value bigger than 0,
literally fucked my performance.*
- `momentum`: Self-explanatory (it is only for "SGD"). Takes float value.
- `norm_ratio`: Gradient clipping ratio. Takes float value.
- `topk`: Tuple value for top-k accuracy calculations (Default: (1, 5)). It is tuple because I c/p related code from Pytorch's
imagenet example without modifying it. It does not have any effects on training, it is for logging/monitoring purposes.
- `print_every_batch_step`: Print loss and accuracy at every x step.
- `save_every_epoch`: Save the model at every epoch.
- `eval_every`: Run the trained model for validation set at every epoch.

### Evaluation Properties

There are 3 properties under this `evaluation_properties`:

- `model_path`: The path for the model file that you want to evaluate.
- `sentence_vocab`: Saved vocabulary (for words) file path.
- `category_vocab`: Saved vocabulary (for labels) file path.

### Model Properties

This is the biggest and longest set of properties. Obviously, it tends to get bigger as long as I add new models. The
root `model_properties` contains several inner dictionaries. The first inner dictionary is `common_model_properties` which
defines the common things that are not change w.r.t. selected learning algorithm. The rest of the inner dictionaries are
the learning algorithms that are developed in this repository.

- `common_model_properties`:
    - `run_mode`: String parameter to define the main executing job. It can be either "train" or "eval_interactive".
    - `use_pretrained_embed`: Boolean parameter to define whether the learning algorithm uses pretrained embeddings or not.
    - `embed_train_type`: String parameter to define whether the embedding layer is trainable or not. It can be "static",
    "nonstatic" or "multichannel". Except Text CNN model, "multichannel" embeddings are not used!
    - `use_batch_norm`: Boolean parameter to determine the batch normalization usage.
    - `batch_norm_momentum`: Float parameter to define batch normalization momentum parameter.
    - `batch_norm_affine`: Boolean parameter to define whether batch normalization uses affine or not.
- `text_cnn`:
    - `use_padded_conv`: Boolean parameter to define whether convolution layer pads the input or not.
    - `dropout_type`: String parameter to choose which dropout method to use. It can be "bernoulli", "gaussian" or
    "variational"
    - `keep_prob`: Float parameter to define the dropout's keeping probability.
    - `filter_count`: Integer parameter to define the filter count in the convolutional layer.
    - `filter_sizes`: List of integers parameter to define the filter sizes in the convolutional layer. Default value is
     [3, 4, 5]. Size of the list is not constant/limited/pre-determined!
- `char_cnn`:
    - `dropout_type`: String parameter to choose which dropout method to use. It can be "bernoulli", "gaussian" or
    "variational"
    - `keep_prob`: Float parameter to define the dropout's keeping probability.
    - `max_sequence_length`: Integer parameter to define the maximum sequence length (in terms of characters). Default value
    is 1014 (as it is defined in CharCNN article).
    - `feature_size`: String parameter to define network size. It can be "large" (conv_filter_count=1024, linear_unit_count=2048),
    "small" (conv_filter_count=256, linear_unit_count=1024) or ""(empty string)
    - `filter_count`: Integer parameter to define filter count in the convolutional layer. If feature_size is empty, all
    convolution layers will use this parameter as the filter_count value.
    - `filter_sizes`: : List of integers parameter to define the filter sizes in the convolutional layers. Default value
    is [7, 7, 3, 3, 3, 3]. Size of the list is limited to 6.
    - `max_pool_kernels`: List of integers parameter to define the kernel sizes in the maxpooling layers. Default value
    is [3, 3, 3]. Size of the list is limited to 3.
    - `linear_unit_count`: Integer parameter to define the number of hidden units in the fully connected layer. If feature_size
    is empty, fully connected layer will use this parameter as the linear_unit_count value.
- `vdcnn`:
    - `keep_prob`: Float parameter to define the dropout's keeping probability. This model only use "bernoulli" dropout.
    - `depth`: Integer parameter to define the depth of the network. It can be 9, 17, 29, or 49.
    - `filter_counts`: List of integers parameter to define the filter counts in the convolutional layers. Default value
    is [64, 128, 256, 512]. Size of the list is limited to 4.
    - `filter_size`: Integer parameter to define the filter size for convolutional layers. All layers use the same size.
    - `use_shortcut`: Boolean parameter to determine shortcut usage in VDCNN model.
    - `downsampling_type`: String parameter to define downsampling method. It can be "resnet", "vgg" or kmax".
    - `maxpool_filter_size`: Integer parameter that defines kernel size for all maxpooling operations.
    - `kmax`: An integer parameter that defines "k" value for KMaxPooling operation.
- `conv_deconv_cnn`:
    - `keep_prob`: Float parameter to define the dropout's keeping probability. This model only use "bernoulli" dropout.
    - `filter_counts`: List of integers parameter to define the filter counts in the encoder convolutional layers. For
    deconvonvolution part, this parameter is reversed. Default value is [300, 600, 500]. Size of the list is limited to 3.
    - `filter_sizes`: List of integers parameter to define the filter sizes in the encoder convolutional layers. For
    deconvonvolution part, this parameter is reversed. Default value is [5, 5, 12]. Size of the list is limited to 3.
    - `strides`: List of integers parameter to define the strides in the encoder convolutional layers. For
    deconvonvolution part, this parameter is reversed. Default value is [2, 2, 1]. Size of the list is limited to 3.
    - `temperature`: Float parameter to define temperature parameter of the Deconvolution stage.
    - `hidden_layer_size`: Integer parameter to define the number of hidden units in the Classifier stage .
- `gru`:
    - `dropout_type`: String parameter to choose which dropout method to use. It can be "bernoulli", "gaussian" or
    "variational"
    - `keep_prob`: Float parameter to define the dropout's keeping probability.
    - `hidden_dim`: Integer parameter to define the hidden dimension.
    - `num_layers`: Integer parameter to define the number of GRU layers.
    - `bidirectional`: Boolean parameter to define bidirectionality.
    - `bias`: Boolean parameter to define the usage of bias.
- `lstm`:
    - `dropout_type`: String parameter to choose which dropout method to use. It can be "bernoulli", "gaussian" or
    "variational"
    - `keep_prob`: Float parameter to define the dropout's keeping probability.
    - `hidden_dim`: Integer parameter to define the hidden dimension.
    - `num_layers`: Integer parameter to define the number of GRU layers.
    - `bidirectional`: Boolean parameter to define bidirectionality.
    - `bias`: Boolean parameter to define the usage of bias.
- `transformer_google`:
    - `use_embed_sqrt_mul`: Boolean parameter to initialize embeddings by multiplying it with the square root of the
    model size. Initially, its value is False.
    - `keep_prob_encoder`: Float parameter to define the dropout's keeping probability in encoder. This model only use
    "bernoulli" dropout.
    - `keep_prob_pe`: Float parameter to define the dropout's keeping probability in positional embeddings. This model
    only use "bernoulli" dropout.
    - `keep_prob_pff`": Float parameter to define the dropout's keeping probability in positional feed-forward. This model
    only use "bernoulli" dropout.
    - `keep_prob_attn`: Float parameter to define the dropout's keeping probability in attention. This model only use
    "bernoulli" dropout.
    - `keep_prob_clf`: Float parameter to define the dropout's keeping probability in classifier. This model only use
    "bernoulli" dropout.
    - `transformer_type`: String parameter to define the job of the transformer model. Currently, it can only take "classifier"
    value.
    - `heads`: Integer parameter to define the number of parallel attention layers.
    - `num_encoder_layers`: Integer parameter to define the number of encoder layers.
    - `num_hidden_pos_ff`: Integer parameter to define number of hidden units in position-wise feed-forward network.
    - `max_length`: Integer parameter to define the maximum length of the input. Default value is 5000.

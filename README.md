# README 

## Intro

Finally, I decided to do my all machine learning, NLP stuff by using PyTorch. The reason is simple actually, I like it more than Tensorflow =) 
Eventually, I won't update the other text_categorization repository, but I will continue to develop same ideas in here. 

## Versions

Before diving into details, the python and library versions are as follows: 

- python 3.6 (works well with 3.7, too)
- pytorch 0.4.1 (no 1.0 preview for windows =))
- torchtext 0.3.1
- gensim 3.4.0 (for fasttext embeddings, as well as OOV Embedding generation. Downgraded from 3.6.0 to 3.4.0(Conda version))

## Code Details

- As the other Tensorflow-based repository, I will use the dataset that me and my old colleagues constructed 3 years ago. "English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset" is publicly available: https://data.mendeley.com/datasets/cdcztymf4k/1
- Initial commit introduces basic Text CNN (from 2014 paper). More models will be added.
- Fasttext embeddings are used (by default but it can be changed). Eventually, one can use Torchtext to download the "pre-defined" embedding files. However, since Turkish embeddings were not included in, I manually edit the Torchtext backend codes (please check the "changes in the torchtext.txt" file). Also note that, everytime you update Torchtext, you need to re-add those changes again.
- Embeddings (whether random or pretrained) can be "static", "nonstatic", or "multichannel".
- For OOV words, OOVEmbeddingCreator is developed (under datahelper/embedding_helper). There are 5 different basic approaches defined to generate OOV embeddings: (1) zeros vector, (2) ones vector, (3) random vector (between 0, 1), (4) (r1, r2) ranged uniformly random vector, (5) Fasttext CharNgram-based vectors.
- Even though I am focusing on Turkish versions of the dataset, I believe "Preprocessor" can work for English dataset, too. In future, I may add more language specific methods. 
- Due to laziness, I defined all my necessary arguments/configs/properties in "main.py". However, I also implemented argparse versions of the same properties but it may lack recent updated properties (Sorry for hard-coded paths). 
- I tested all training, evaluation, model/vocabulary saving/loading aspects of the code for several epochs without any problem (except out of memory errors =)).

## To-do 

- [x] ~~Variational Dropout. Update: Variational and Gaussian dropout methods are added. Reference: [Variational Dropout and
the Local Reparameterization Trick](https://arxiv.org/pdf/1506.02557.pdf)~~
- [x] ~~Extend main flow and learning models with respect to new dropout models.~~ 
- [x] ~~Add character-level data preprocessing.~~
- [x] ~~Add character-level data loading.~~
- [ ] Run the current piece of code for the aforementioned datasets and define a text categorization baseline (for both Turkish and English). 
- [ ] Variational Dropout related extensions (current version is from 2015 paper but obviously more recent versions are out there for me to implement =)) + bayes by backprop for CNN (a.k.a. Bayesian CNN)
- [ ] Attention.
- [ ] Different learning algorithms (DeepCNN, LSTM, GRU, any-kind-of-hybrid versions of those algorithms, transformers).
  - [x] TextCNN
  - [x] GRU 
  - [x] LSTM
  - [x] ~~Multilayer CNN~~ (I removed this model and decided to continue with CharCNN and VDCNN instead).
  - [x] CharCNN
  - [x] VDCNN (Very Deep CNN)
  - [ ] Transformers
  - [ ] Conv-Deconv CNN
  - [ ] Encoder-Decoder GRU
  - [ ] Encoder-Decoder LSTM
  - [ ] Hybrid stuff (Like CNN+LSTM/GRU)
- [ ] CRF layer to be able to do NER experiments.
- [ ] For Turkish, I plan to add morphological disambiguation (https://github.com/erayyildiz/Neural-Morphological-Disambiguation-for-Turkish). 
- [ ] Different language models.
  - [ ] ELMO (pretrained Turkish/English embeddings)
  - [ ] BERT (pretrained Turkish/English embeddings)
- [ ] Document length categorization/NER tasks (Current runs are on sentence-level; however, with ELMO/BERT I will be able to do document level tasks, too.)

## Project Skeleton

I try to keep every part of the project clean and easy to follow. Even though the folders are self explanatory for me, let me explain them for those who may have hard time to understand.

- `./argument/argument_reader.py` contains arguments and their parsers. Default values are same with the hard coded variables in 
`main.py`
- `./datahelper/dataset_reader.py` contains the "DatasetLoader" object that reads a text dataset, splits it into 3 subsets (train/vali/test), creates vocabulary and iterators. It is a little bit hard-coded for the dataset I am using now. However, it is easy to make changes to use it for your own dataset.
- `./datahelper/embedding_helper.py` is a helper class to generate OOV word embeddings. To use Fasttext-based OOV embedding generation, it leverages Gensim!
- `./datahelper/preprocessor.py` contains the "Preprocessor" object and actions to apply on sentences. 
- `./dropout_models/gaussian_dropout.py` contains the Gaussian Dropout object. 
- `./dropout_models/variational_dropout.py` contains the Variational Dropout object. 
- `./dropout_models/dropout.py` contains the Dropout object which you can select your dropout type among Bernoulli (basic), Gaussian and Variational dropout types. 
- `./evaluation/evaluate.py` contains two methods for evaluation. The first one is evaluating validation and/or test sets while training. The other method is for interactive evaluation. Note that you need "spacy" to tokenize test sentences for interactive evaluation (note that my original dataset is already tokenized, so you do not need to use spacy while training).
- `./model/xyz.py` contains network objects.
- `./model/Util_xyz.py` contains custom-defined objects that are used in `xyz`.
- `./training/train.py` contains training specific methods. 
- `./utils/utils.py` contains both utility and common methods that are being used in several places in the project.
- `./main.py` is the main code. Run arguments/parameters/configurations are at the top of this file.

## How-to-run

### Important Note Before Start

I had to make some changes in the torchtext backend codes to be able to do several stuffs:

- I don't know why, torchtext does not split a dataset into 3 subsets (train/val/test) even if there is a function for it. I changed it to fix that issue. Hopefully, one day torchtext will fix it offically =)
- To be able to work with Turkish Fasttext embeddings, I added its respective alias.
- To be able to apply Fasttext's CharNGram to OOV words to generate OOV embeddings, a minor change has been made to Vector object.

### Run Arguments

There are 3 dictionaries defined to hold run arguments. 

- `dataset_properties` holds dataset related arguments:
  - stop_word_path: The file you keep your language's stop words
  - data_path: The original dataset file
  - embedding_vector: Embedding alias that torchtext needs/uses while building vocabulary (predefined aliases can be found torchtext's vocab.py file).
  - vector_cache: The embedding file that torchtext creates the first time it runs with the defined embedding. To prevent it to download same file over and over again, you need to provide its path.
  - pretrained_embedding_path: This is the original, Gensim readable, embedding files (note that only use case for this is Fasttext-based OOV word generation).
  - checkpoint_path: The path for saved model file that you want to continue your training.
  - oov_embedding_type: It can be "zeros", "ones", "random", "uniform" or "fasttext_oov" and specifies which method to use to generate OOV word vectors.
  - batch_size: Self-explanatory.
  
 - `model_properties` holds model/algorithm-related arguments:
   - use_pretrained_embed: Can be "True" if you want to use known word embedding models, or "False" if you want to use random vectors.
   - embed_train_type: Can be "static" if you want non-trainable, "nonstatic" if you want trainable or "multichannel" if you want multichannel embeddings as your inputs.
   - use_padded_conv: A boolean argument that specifies whether convolution filters apply padding or not.
   - dropout_type: Dropout type. Can be "bernoulli", "gaussian" or "variational".
   - keep_prob: Dropout probability.
   - use_batch_norm: A boolean argument to use batch normalization.
   - batch_norm_momentum: Batch normalization's momentum parameter.
   - batch_norm_affine: Batch normalization's affine parameter.
   - rnn_hidden_dim: Hidden dimension for RNN/GRU/LSTM-based models.
   - rnn_num_layers: Number of layers for RNN/GRU/LSTM-based models. 
   - rnn_bidirectional: A boolean argument for RNN/GRU/LSTM-based models to define bidirectionality.
   - rnn_bias: A boolean argument for RNN/GRU/LSTM-based models to use bias.
   - filter_count: Number of filters for CNN-based models.
   - filter_sizes: List of convolution filter sizes for CNN-based models(Example: [3, 4, 5]).
   - max_sequence_length: An integer parameter for char-level CNN models. Initial value is set to 1014 (same as the original papers) 
   - feature_size: Network size parameter for CharCNN. It can be "large" (conv_filter_size=1024, linear_unit_count=2048), "small" (conv_filter_size=256, linear_unit_count=1024) or ""(empty string). 
   - charcnn_filter_count: If feature_size is empty, all convolution layers will use this parameter as it is filter_count parameter.
   - charcnn_filter_sizes: If feature size is empty, all convolution layers will use this parameter as it is filter_size parameter. It contains list of integer and list's length is equal to number of convolution layers.
   - max_pool_kernels: List of max_pooling operations' kernel sizes. Initial list contains 3 integer values since the CharCNN architecture has 3 max_pooling operations.
   - linear_unit_count: If feature size is empty, all linear units will use this parameter as it is their in and/or out feature size.
   - depth: An integer parameter that defines the depth of a VDCNN model. It can take only 9, 17, 29, 49 as value.
   - vdcnn_filter_counts: List of integers that defines the out channels of all convolutional layers.
   - vdcnn_filter_size: An integer parameter that defines the kernel_size for all convolution operations.
   - use_shortcut: A boolean parameter that defines whether using shortcut operation in VDCNN.
   - downsampling_type: A string parameter that defines downsampling operation type. It can be "resnet", "vgg" or kmax".
   - maxpool_kernel_size: An integer parameter that defines kernel size for all maxpooling operations.
   - kmax: An integer parameter that defines "k" value for KMaxPooling operation. 
   - run_mode: Can be "train" to start training process or "eval_interactive" to test your saved model(s) interactively. 
  
 - `training_properties` holds training-related arguments:
   - learner: Argument to choose which learning algorithm to use. It can be "textcnn", "gru", "lstm", "charcnn", and "vdcnn" (Update: 29 Nov 2018) 
   - optimizer: It can be either "Adam" or "SGD".
   - learning_rate: Self-explanatory.
   - weight_decay: L2 normalization term. Note that for my case, any value bigger than 0, literally fucked my performance. 
   - momentum: Self-explanatory (note that if you use "Adam" it will be ignored, it is only for "SGD").
   - norm_ratio: Gradient clipping ratio.
   - topk: Tuple value for top-k accuracy calculations (Default: (1, 5)). It is tuple because I c/p related code from Pytorch's imagenet example without modifying it.  
   - print_every_batch_step: Print loss and accuracy at every x step.
   - save_every_epoch: Save the model at every epoch.
   - eval_every: Run the trained model for validation set at every epoch.
 
  - `evaluation_properties` holds interactive evaluation related arguments:
    - model_path: The path for the model file that you want to evaluate.
    - sentence_vocab: Saved vocabulary (for words) file path.
    - category_vocab: Saved vocabulary (for labels) file path.

### Training/Evaluation

After you make the necessary changes in "changes_in_torchtext" and edit the hard coded paths in "main.py", it should not be a problem to start your own training/evaluation. 
If you succesfully train and save a model, you can evaluate the saved model interactively by changing the "run_mode" parameter from "train" to "evaluate_interactive". 

## Results

This section presents the Top-1 and Top-5 test accuracies for **text categorization task** of my experiments. Due to computational resource limit, I cannot test every single parameter/hyperparameter. In general, I hold algorithm parameters same for all experiments; however, I change embedding related parameters. I assume the result table is self-explanatory. As a final note, I *won't* share my best models and I *won't* guarantee reproducibility. Dataset splits (training/validation/test) are deterministic for all experiments, but anything else that needs random initialization is non-deterministic. 

Note: Epoch is set to 20 for all experiments, until further notice (last update: 31-10-2018). However, if I believe that results may improve, I let the experiment run for 10 more epochs (at most 30 epoch per experiments). 

### Test Results of 1-Layer CNN + FC (TextCNN)

| Language | # Of Categories | Pre-trained Embedding | OOV Embedding | Embedding Training | Top-1 Test Accuracy | Top-5 Test Accuracy |   
|----------|:-----------------------------:|-----------------------|---------------|--------------------|:-------------------:|:-------------------:|
|Turkish|25| Fasttext | zeros | static	| 49.4565 | 76.2760 |
|Turkish|25| Fasttext | zeros | nonstatic	| 62.6054 | 86.3384 |
|Turkish|25| Fasttext | Fasttext | static	|  49.6810  | 75.2684 |
|Turkish|25| Fasttext | Fasttext | nonstatic	| 63.9391  | 87.9597 |
|Turkish|49| Fasttext | zeros | static	| 43.5519  | 68.4336 |
|Turkish|49| Fasttext | zeros | nonstatic	| 56.0081  | 79.8634 |
|Turkish|49| Fasttext | Fasttext | static	| 43.8025  | 68.8641 |
|Turkish|49| Fasttext | Fasttext | nonstatic	| NaN (Training)  | NaN (Training) |
|English|25| Fasttext | zeros | static	| NaN (TBA) | NaN (TBA) |
|English|25| Fasttext | zeros | nonstatic	| NaN (TBA) | NaN (TBA) |
|English|25| Fasttext | Fasttext | static	|  NaN (TBA)  | NaN (TBA) |
|English|25| Fasttext | Fasttext | nonstatic	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | zeros | static	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | zeros | nonstatic	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | Fasttext | static	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | Fasttext | nonstatic	| NaN (TBA)  | NaN (TBA) |


### References for code development: 
Below repositories really helped me to write a decent and working code:
- https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch
- https://github.com/bentrevett/pytorch-sentiment-analysis
- https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
- https://github.com/felix-laumann/Bayesian_CNN/
- https://github.com/kefirski/variational_dropout/
- https://github.com/dreamgonfly/deep-text-classification-pytorch/

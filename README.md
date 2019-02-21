# README 

## Update 21-02-2019
 
- Precision, recall and F1 metrics are added into "ner_scorer.py". 
  - Since these metrics must be calculated for full set (not batch-based), I changed the evaluator flow a little bit.
  - Evaluator reports mean precision, recall and F1 scores over all tags/named-entities. 
  - Detailed, tag-based, scores can be also reported by activating boolean detailed_ner_log (default value is true).
- In LSTM, I encountered a minor bug while using "bidirectional=true". Hopefully, it is fixed (at least training/evaluating was working on a small set).
- I tried a larger set to see whether my code is working but I got a "cuda illegal memory access" error. I think it is because OOM issues, but I am not sure for now. 
- An "allowed_transition" stuff will be added in near future (like allennlp/conditional_random_field.py).
- Also, I updated my libraries, hence requirements.txt is changed, too =)
- Personal life issues still continues, hence slow development-slow experiment mode still continues.
 
# Table Of Contents

- [Introduction](#introduction)
- [Library Dependencies](#library-dependencies)
- [Project Skeleton](#project-skeleton)
- [Project Details](#project-details)
- [To-do](#to-do)
- [How-to-run](#how-to-run)
  - [Important Note Before Start](#important-note-before-start)
  - [Configuration JSON Format](#configuration-json-format)
  - [How to Run Main](#how-to-run-main)
  - [Training from Scratch-Training from Checkpoint-Interactive Evaluation](#training-from-scratch-training-from-checkpoint-interactive-evaluation)
- [Results](#results)
  - [Test Results for TextCNN](#test-results-for-textcnn)
- [Previous Updates](#previous-updates)
  - [January 2019 - Wiki Link](https://github.com/hbahadirsahin/nlp-experiments-in-pytorch/wiki/Previous-Updates-(January-2019))
  - [February 2019](#february-2019)
- [References for Code Development](#references-for-code-development)

## Introduction

This is my personal, pet project which I apply machine learning and natural language processing stuffs by using PyTorch. I stopped working with Tensorflow after some hellish times that I could not do some basic extentions (such fasttext based oov embeddings, details are below). Also, Tensorflow's updates and functionality deprecation rate is annoying for me. 

In this repository, I implement popular learning models and extend them with different minor adjustments (like variational dropouts). Even though it is really slow, I execute experiments by using these models on a dataset which me and my old colleagues in Huawei constructed (details are below, again) and try to announce experiment results.

## Library Dependencies

Before diving into details, the python and library versions are as follows: 

- python 3.6 (works well with 3.7, too)
- torch 1.0.0
- torchtext 0.3.1
- numpy 1.15.4 (due to PyTorch 1.0)
- setuptools 40.6.2 (Hell no idea why pipreqs put this into requirements.txt)
- spacy 2.0.16 (for interactive evaluation only)
- gensim 3.6.0 (for fasttext embeddings, as well as OOV Embedding generation.)

## Project Skeleton

I try to keep every part of the project clean and easy to follow. Even though the folders are self explanatory for me, let me explain them for those who may have hard time to understand.

- `./crf/CRF.py` contains the conditional random field implementation (not finished yet). 
- `./datahelper/dataset_reader.py` contains the "DatasetLoader" object that reads a text dataset, splits it into 3 subsets (train/vali/test), creates vocabulary and iterators. It is a little bit hard-coded for the dataset I am using now. However, it is easy to make changes to use it for your own dataset.
- `./datahelper/embedding_helper.py` is a helper class to generate OOV word embeddings. To use Fasttext-based OOV embedding generation, it leverages Gensim!
- `./datahelper/preprocessor.py` contains the "Preprocessor" object and actions to apply on sentences. 
- `./dropout_models/gaussian_dropout.py` contains the Gaussian Dropout object. 
- `./dropout_models/variational_dropout.py` contains the Variational Dropout object. 
- `./dropout_models/dropout.py` contains the Dropout object which you can select your dropout type among Bernoulli (basic), Gaussian and Variational dropout types. 
- `./evaluation/evaluator.py` is the factory for evaluation objects that are used in model trainings as well as interactive evaluation.
- `./evaluation/xyz_evaluator.py` methods are the evaluator functions for specified models.
- `./model/xyz.py` contains network objects.
- `./model/Util_xyz.py` contains custom-defined objects that are used in `xyz`.
- `./optimizer/custom_optimizer.py` contains custom-defined optimizer objects.
- `./scorer/accuracy_scorer.py` contains classification accuracy metric calculations.
- `./scorer/ner_scorer.py` contains NER-task related metric calculations.
- `./training/trainer.py` is a class that returns the necessary trainer for the user's selected learning model
- `./training/xyz_trainer.py` methods are the trainer functions for specified models.
- `./utils/utils.py` contains both utility and common methods that are being used in several places in the project.
- `./main.py` is the main code. To execute this project, one needs to provide a valid `config.json` file which contains the necessary configuration properties.
- `./config/config.json` is the configuration file.  

## Project Details

- As the other Tensorflow-based repository, I will use the dataset that me and my old colleagues constructed 3 years ago. "English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset" is publicly available: https://data.mendeley.com/datasets/cdcztymf4k/1
- Text CNN, CharCNN, VDCNN, Conv-Deconv CNN, basic LSTM/GRU and Transformer (Google version) models are currently available to train and evaluate in the repository. More models will be added. 
- Fasttext embeddings are used (by default but it can be changed). Eventually, one can use Torchtext to download the "pre-defined" embedding files. However, since Turkish embeddings were not included in, I manually edit the Torchtext backend codes (please check the "changes in the torchtext.txt" file). Also note that, everytime you update Torchtext, you need to re-add those changes again.
- Embeddings (whether random or pretrained) can be "static", "nonstatic", or "multichannel".
- For OOV words, OOVEmbeddingCreator is developed (under datahelper/embedding_helper). There are 5 different basic approaches defined to generate OOV embeddings: (1) zeros vector, (2) ones vector, (3) random vector (between 0, 1), (4) (r1, r2) ranged uniformly random vector, (5) Fasttext CharNgram-based vectors.
- Even though I am focusing on Turkish versions of the dataset, I believe "Preprocessor" can work for English dataset, too. In future, I may add more language specific methods. 
- Main code loads properties from config.json (inside config folder). 
- I tested all training, evaluation, model/vocabulary saving/loading aspects of the code for several epochs without any problem (except out of memory errors =)).

## To-do 

- [x] ~~Better configuration/property reading, handling, instead of hard-coded dictionaries~~ (Update: 11-Jan-2019)
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
  - [x] Transformer (*Attention is All You Need* version) (**Modified for Text Classification/NER!**) 
  - [ ] Transformer (*Improving Language Understanding by Generative Pre-Training* version)
  - [ ] Transformer-XL (*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context* version)
  - [x] Conv-Deconv CNN
  - [ ] Encoder-Decoder GRU
  - [ ] Encoder-Decoder LSTM
  - [ ] Hybrid stuff (Like CNN+LSTM/GRU)
- [x] ~~CRF layer to be able to do NER experiments~~.
  - [ ] Add new models that will use CRF as their last layer (such as LSTMCRF, GRUCRF, CNNCRF, etc.)
  - [ ] Develop NER-related performance metrics and update training/evaluation flows to use these metrics.
- [ ] For Turkish, I plan to add morphological disambiguation (https://github.com/erayyildiz/Neural-Morphological-Disambiguation-for-Turkish). 
- [ ] Different language models.
  - [ ] ELMO (pretrained Turkish/English embeddings)
  - [ ] BERT (pretrained Turkish/English embeddings)
- [ ] Document length categorization/NER support (Conv-Deconv CNN implementation supports document-length tasks, but more support will come with ELMO and BERT update).

## How-to-run

### Important Note Before Start

I had to make some changes in the torchtext backend codes to be able to do several stuffs:

- I don't know why, torchtext does not split a dataset into 3 subsets (train/val/test) even if there is a function for it. I changed it to fix that issue. Hopefully, one day torchtext will fix it offically =)
- To be able to work with Turkish Fasttext embeddings, I added its respective alias.
- To be able to apply Fasttext's CharNGram to OOV words to generate OOV embeddings, a minor change has been made to Vector object.
- To be able to read any dataset without any problem, a minor change has been made to torchtext's utils.py.

### Configuration JSON Format

To be able to run the main code, you need to provide a valid JSON file which contains 4 main properties. These are `dataset_properties`, `model_properties`, `training_properties`, and `evaluation_properties`:

- `dataset_properties` contains dataset-related information such as path, embedding, batch information.
- `model_properties` contains model-related parameters. Inside this property,
  - `common_model_properties` contains common properties for all models like embeddings, vocabulary size, etc.
  - `model_name` (like text_cnn, char_cnn, etc.) contains model-specific properties.
- `training_properties` contains training-related properties.
- `evaluation_properties` contains evaluation-related properties.

Details of the `config.json` can be found in "/config/README.md" folder.

### How to Run Main

If you make the necessary changes described in "changes in torchtext.txt" and prepare "config.json", you have two ways to run the code.

- If you are using an IDE, copy/paste your "config.json" file's path as an argument and press run button.
- If you are an old-school command window lover, type `python main.py --config /path/to/config.json`.

### Training from Scratch-Training from Checkpoint-Interactive Evaluation

You can train your model from 0th epoch until max_epoch, and/or continue your training from xth epoch to the end. You do not need to do anything extra for the first case; however, to be able to continue your training you need to make necessary changes in "config.json":

- If `dataset_properties/checkpoint_path` is empty, the code will start a new training process. If you type your saved PyTorch model, the main flow will automatically load it and continue from where it left.
  - Additionally, you can provide saved vocabulary files for words (`dataset_properties/saved_sentence_vocab` (don't ask why it is sentence)) and labels (`dataset_properties/saved_category_vocab`).
  
To be able to activate interactive evaluation, you need to make necessary changes in "config.json":

- Change `model_properties/common_model_properties/run_mode`'s value to "eval_interactive".
- Provide your model's path to be evaluated and your saved vocabulary files' path by using `evaluation_properties`.

## Results

This section presents the Top-1 and Top-5 test accuracies for **text categorization task** of my experiments. Due to computational resource limit, I cannot test every single parameter/hyperparameter. In general, I hold algorithm parameters same for all experiments; however, I change embedding related parameters. I assume the result table is self-explanatory. As a final note, I *won't* share my best models and I *won't* guarantee reproducibility. Dataset splits (training/validation/test) are deterministic for all experiments, but anything else that needs random initialization is non-deterministic. 

Note: Epoch is set to 20 for all experiments, until further notice (last update: 31-10-2018). However, if I believe that results may improve, I let the experiment run for 10 more epochs (at most 30 epoch per experiments). 

Note 2 (**Update: 22-01-2019**): Most of the English-language experiments are executed in Google Cloud (by using 300$ initial credit). Since, I want to finish as many experiments as possible, I cannot increase the max_epoch from 20 to 30. In this experiments, I saw that validation loss and accuracies were improving in every epoch until the 20th, and I am pretty sure models can improve further. Unfortunately, I chose the maximum number of experiment runs instead of best results for each experiment in this trade-off. 

### Test Results for TextCNN

|#| Language | # Of Categories | Pre-trained Embedding | OOV Embedding | Embedding Training | Top-1 Test Accuracy | Top-5 Test Accuracy |   
|-|:--------:|:-----------------------------:|-----------------------|---------------|--------------------|:-------------------:|:-------------------:|
|1|Turkish|25| Fasttext | zeros | static	| 49.4565 | 76.2760 |
|2|Turkish|25| Fasttext | zeros | nonstatic	| 62.6054 | 86.3384 |
|3|Turkish|25| Fasttext | Fasttext | static	|  49.6810  | 75.2684 |
|4|Turkish|25| Fasttext | Fasttext | nonstatic	| 63.9391  | 87.9597 |
|5|Turkish|49| Fasttext | zeros | static	| 43.5519  | 68.4336 |
|6|Turkish|49| Fasttext | zeros | nonstatic	| 56.0081  | 79.8634 |
|7|Turkish|49| Fasttext | Fasttext | static	| 43.8025  | 68.8641 |
|8|Turkish|49| Fasttext | Fasttext | nonstatic	| 60.4009  | 82.7879 |
|9|English|25| Fasttext | zeros | static	| 56.2290 | 83.2425 |
|10|English|25| Fasttext | zeros | nonstatic	| 64.2642 | 89.2115 |
|11|English|25| Fasttext | Fasttext | static	| 56.5313 | 83.9873 |
|12|English|25| Fasttext | Fasttext | nonstatic	| 65.9558 | 91.1536 |
|13|English|49| Fasttext | zeros | static	| 51.3862 | 78.7806 |
|14|English|49| Fasttext | zeros | nonstatic	| 59.2086*  | 84.8054 |
|15|English|49| Fasttext | Fasttext | static	| 51.7878 | 79.9472 |
|16|English|49| Fasttext | Fasttext | nonstatic	| 55.3833*  | 80.4958 |

* Note that the experiment 14 resulted with a better score than 16, unlike other similar setups. The main reason is, I changed the "learning_rate" of the optimizer to a smaller value for the experiment 16 (well, for the sake of the experiment =)), and it appears that smaller learning rate made the learning process a bit slower (in terms of number of epochs). If I can find a chance to run this experiment again in Google Cloud (a.k.a. have enough credit to run it one more time), I will update the learning rate properly. 

## Previous Updates

In this title, I will save the previous updates for me and the visitors to keep track.

## February 2019

### 04-02-2019

- Code development for NER evaluation (F1 scores for tag based, bio+tag based) still continues. I slowed down it a little bit due to personal life issues.
- Decided to simplify README.md and use [Wiki](https://github.com/hbahadirsahin/nlp-experiments-in-pytorch/wiki) of this repository.
  - All entries related to the updates in January 2019 are moved to [related Wiki page](https://github.com/hbahadirsahin/nlp-experiments-in-pytorch/wiki/Previous-Updates-(January-2019)). 
  - I will save my experiment results in Wiki, too.
  - While trying to figure out how to use README and Wiki more efficiently, I will figure better things out hopefully =)
- As you may noticed, TextCNN experiments are finished. I will continue with LSTM/GRU experiments (first, I have to figure out a good parameter set =)).

### Update 12-02-2019
 
- Since I separated classification and NER trainers/evaluators, I decided to create "scorer" folder to prevent bloating "utils.py" with metric calculation functions.
 - "scorer/" will contain current and future metric calculation methods (not giving too much details, since you can always check it =)).
- I encountered some bugs in NER training and evaluation processes due to save/load functionalities. Hopefully, I fully fixed them. but if anyone out there reading this and using this repository, if you find any bugs, just let me know.
- Made some minor changes in namings and indexing (not much crucial stuff, details can be found in git commit message).
- Personal life issues still continues, hence slow development-slow experiment mode still continues.

## References for Code Development

Below repositories really helped me to write a decent and working code:
- https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch
- https://github.com/bentrevett/pytorch-sentiment-analysis
- https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
- https://github.com/felix-laumann/Bayesian_CNN/
- https://github.com/kefirski/variational_dropout/
- https://github.com/dreamgonfly/deep-text-classification-pytorch/
- https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
- https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
- https://github.com/threelittlemonkeys/lstm-crf-pytorch/
- https://github.com/ymym3412/textcnn-conv-deconv-pytorch/blob/master/model.py
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://github.com/huggingface/pytorch-openai-transformer-lm

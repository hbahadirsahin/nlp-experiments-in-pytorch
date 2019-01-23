# README 

## Update 23-01-2019

- Conditional Random Field (CRF) class is added into the project. I have not tested it yet. So, I am pretty sure it has lots of bugs =) (wait for the future updates).
  - A new property "training_properties/task" is defined in config.json. Details are in "/config/README.md"
  - Dataset reader code is updated to handle NER datasets. Previous version was reading the sentence and category columns of the dataset while ignoring ner column. Now, it reads NER column, assigns the column to the respective field, and builds NER vocabulary, if the "task" property is "ner".
  - Eventually, I made some changes in main.py. I added CRF into the model creation method, but it is for testing. I don't have any plans to keep it there. Depending on the given "task", NER-counterparts of the category-related actions in main are added.
- Again, CRF is not tested! In near future, I will spend some time on doing basic tests to idenfity bugs, missings and improvement possibilities.  
- First, but not last, batch of bugfixes have been pushed.
  - All problematic things related to DatasetLoader have been fixed (Check this [commit message](https://github.com/hbahadirsahin/nlp-experiments-in-pytorch/commit/1b66f424b59048245b3f046295590388d49cddca) for details).
  
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
  - [January 2019](#january-2019)
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
- [ ] CRF layer to be able to do NER experiments.
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

| Language | # Of Categories | Pre-trained Embedding | OOV Embedding | Embedding Training | Top-1 Test Accuracy | Top-5 Test Accuracy |   
|----------|:-----------------------------:|-----------------------|---------------|--------------------|:-------------------:|:-------------------:|
|Turkish|25| Fasttext | zeros | static	| 49.4565 | 76.2760 |
|Turkish|25| Fasttext | zeros | nonstatic	| 62.6054 | 86.3384 |
|Turkish|25| Fasttext | Fasttext | static	|  49.6810  | 75.2684 |
|Turkish|25| Fasttext | Fasttext | nonstatic	| 63.9391  | 87.9597 |
|Turkish|49| Fasttext | zeros | static	| 43.5519  | 68.4336 |
|Turkish|49| Fasttext | zeros | nonstatic	| 56.0081  | 79.8634 |
|Turkish|49| Fasttext | Fasttext | static	| 43.8025  | 68.8641 |
|Turkish|49| Fasttext | Fasttext | nonstatic	| 60.4009  | 82.7879 |
|English|25| Fasttext | zeros | static	| 56.2290 | 83.2425 |
|English|25| Fasttext | zeros | nonstatic	| 64.2642 | 89.2115 |
|English|25| Fasttext | Fasttext | static	| 56.5313 | 83.9873 |
|English|25| Fasttext | Fasttext | nonstatic	| 65.9558 | 91.1536 |
|English|49| Fasttext | zeros | static	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | zeros | nonstatic	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | Fasttext | static	| NaN (TBA)  | NaN (TBA) |
|English|49| Fasttext | Fasttext | nonstatic	| 55.3833  | 80.4958 |

## Previous Updates

In this title, I will save the previous updates for me and the visitors to keep track.

### January 2019

#### 21-01-2019

- All print-oriented logs are converted to logging library-based loggers.
- `/config/config.logger` file is added as a logger configuration file.
- README.md changes
  - Table of contents added.
  - Format changes (title revisions, section replacements, etc.).

#### 20-01-2019

- Thanks to Tesla V100, I got the latest experiment results in 20 hours (yay!). 
- I find out that "Padam" optimizer works flawless w.r.t. usual Adam. It is more robust through each step and have not encountered any weird, numerical problems (which I've seen a lot while using Adam). So, if you are reading this and forking/copy-pasting this library to train your own models, I strongly suggest you to use Padam as your optimizer.
- I do not have any development/fix updates. 
  - However, I am working on CRF and plug-in/out CRF-Layer codes (Did I mention I hate CRF?).
  - Also, replacing "print()" oriented logs with "logging" library.

#### 19-01-2019

- Finally, I got another test score (it took 1 month to finish 20 epoch in a workstation-strong CPU =)). 
- Currently, I have no development and/or fix update. 
- Instead, I am trying to find a solution for my resource bottleneck. In last 3 days, I was struggling to understand Google Cloud and its compute engine for my mental goodness. After 3 painful, soul-crashing days (GPU quota problem, GPU quota ticket problem, ssh problem, python problem, library problem, pip problem, fucking no module "xyz" is found problem), I could start a training in a machine with Tesla V100 (every poor human being's dream card).
  - Hopefully, by opening lots of new google accounts (to leverage initial $300 credit, until my unique credit cards diminish), I will be able to get several test results faster.  

#### 16-01-2019

- I added two new properties to `config.json/dataset_properties` (min_freq and fixed_length) to reduce memory consumption. You are still able to use dynamic input size and assign every seen word in your vocabulary if you have enough memory. Check `config/README.md` for detailed information.
- Sadly, I encountered the worst problem in PyTorch related to CUDA OOM error, which is model reloading increases the memory consumption =/ In short, I could start a training process (English dataset/non-static/zeroes oov/text_cnn) and it iterated for 2 epochs without any problem (stable memory consumption with 1.5GB of free GPU memory). Then, I saved the model to continue the process later. However, after I loaded the model, the code directly raised CUDA OOM error. I tried to apply things that I've read in PyTorch's forums; however, those so called fixes did not help me. Things that I've found and tried:
  - I tried to delete the checkpoint reference after model loading (https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213)
  - I tried to catch OOM error and free some memory after it (https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781/2?u=ptrblck)
- In conclusion, if you have a spare computer that can do your training until the end, I am %100 sure that this repository does not have memory leak. As long as your input and model sizes are reasonable, it will train. However, if you do not have such a luxury, I can't do anything about it. But if you have any suggestions, I'd be really happy to listen/apply =) 

#### 15-01-2019

- I created a README for the config.json. It can be found in newly created config folder.
- Last night, I did some research, basic math (to calculate model size) and experiments about possible memory leaks to prevent CUDA OOM errors. Basically, I could not find any memory leak in normal memory and GPU memory. In conclusion, my model (for English) is too big to train in my own GPU. 
  - Eventually, I did not want to play with model parameters to reduce the size, but I decided to reduce it by dataset level.
  - I have not fixed any sentence length and used all words in my vocabularies (min_freq=1). In Turkish experiments, since the dataset is not big, I did not face any problems, its a total different story in English. 
  - I am currently testing the fixed_length and min_freq parameters to control my model size. Until now, tests are going well. Depending on the results, I will put this two parameters into the config.json.

#### 14-01-2019

- After I find out vocabulary caching has bugs and could not fix it, I removed vocabulary caching functionality from code (both save/load parts). 
  - Even though saving is not a problem, to be able to load a Vocab object, one needs to do too much workaround. I wasted my 6 hours to make it work, but no chance (Vocab objects can be loaded by pickle, but all dataset iterators also want to hold a Vocab object inside which can be done by using `build_vocab()` method in normal dataset reading process. If one loads external, cached vocabularies, you jump this step and can't feed these iterators with vocab objects, a.k.a. can't train due to missing Vocab objects in iterator).
  - I will wait for torchtext to provide native support to vocabulary saving/loading.
- I will spend some time on monitoring and optimizing my models/training flows for GPU memory optimization. In my laptop, I am bounded with 3GB GPU memory, and I cannot train big models (I have to say that I did not face such problems in Tensorflow for same model/dataset/parameter sets)

#### 13-01-2019

- Final fixes are applied in transformer model, and it is trainable.
- However, depending on the parameters and model size, it can produce CUDA OOM (out of memory) error pretty easily.
  - Related to the memory error, somehow PyTorch seems can't handle CUDA memory as good as Tensorflow. I will do some research about it to optimize GPU memory in the following days (using `torch.cuda.empty_cache()` for this purpose in training steps isn't enough).
- There are some minor updates in training process (both in single and multiple trainers).
  - Since NoamOptimizer does not inherit the PyTorch optimization, I put checkers into the trainers for this optimizer whenever ".zero_grad()", ".step()", ".save()" and ".load()" functions are called for the optimization object.
- A new optimizer is added into custom_optimizer: "Padam". The reference paper is [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/pdf/1806.06763.pdf).
  - Yesterday, I was reading reddit/ML about Adam-related problems and saw this paper. I have not tested it, in terms of optimality/training-test results, but I will give it a shot.

#### 12-01-2019

- I started to work on transformer_google model. Obviously, it cannot be trained by its current version.
- I have fixed several major bugs. 
  - Classifier block's keep_prob parameter was missing. Hence, it is added to config.json as well as the model flow.
  - Nobody told me that in MultiHeadedAttention, model dimension should be divisible by the number of heads (attention layers). This lack of knowledge costed me 2 hours, but it is fixed (and will be checked inside the model).
- Tests are going on (not unit tests obviously)
- README.MD changes.
- MIT Licence is added.

#### 11-01-2019

I stopped being a lazy guy and changed the current code execution stuff:

- All hard-coded, property holding dictionaries inside main.py are removed.
- Instead, a "config.json" file is created and the main code will ask this file's path (as argument) from you to run the project, properly. 
- Detailed description of this file will be added into this readme (but until I write it, you can always open the file. Believe me, it is not too complicated =)). 
- With respect to new kind of property handling, I changed every related variable/argument initialization in the main and model files. 
- ~~A complete README.MD overhaul is coming on its way~~. (Done!)
- Still, I have not tested Transformer code. Don't be mad at me if you c/p it and can't get results for your homework(s) =)
- Tests are really really slow in CPU workstation and I still play games in my daily-life computer instead of running experiments.

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
- https://github.com/ymym3412/textcnn-conv-deconv-pytorch/blob/master/model.py
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://github.com/huggingface/pytorch-openai-transformer-lm

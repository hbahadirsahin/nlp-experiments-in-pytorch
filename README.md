# README 

## Intro

Finally, I decided to do my all machine learning, NLP stuff by using PyTorch. The reason is simple actually, I like it more than Tensorflow =) 
Eventually, I won't update the other text_categorization repository, but I will continue to develop same ideas in here. 

## Versions

Before diving into details, the python and library versions are as follows: 

- python 3.6
- pytorch 0.4.1 (no 1.0 preview for windows =))
- torchtext 0.3.1
- gensim 3.6.0 (for fasttext embeddings, as well as OOV Embedding generation. More details below)

## Code Details

- As the other Tensorflow-based repository, I will use the dataset that me and my old colleagues constructed 3 years ago. "English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset" is publicly available: https://data.mendeley.com/datasets/cdcztymf4k/1
- Initial commit introduces basic Text CNN (from 2014 paper). 
- Fasttext embeddings are used (by default but it can be changed). Eventually, one can use Torchtext to download the "pre-defined" embedding files. However, since Turkish embeddings were not included in, I manually edit the Torchtext backend codes (please check the "changes in the torchtext.txt" file). Also note that, everytime you update Torchtext, you need to re-add those changes again.
- Embeddings (whether random or pretrained) can be "static", "nonstatic", or "multichannel".
- For OOV words, OOVEmbeddingCreator is defined (under datahelper/embedding_helper). There are 5 different basic approaches are defined to create OOV embeddings: (1) zeros vector, (2) ones vector, (3) [0,1) ranged random vector, (4) (r1, r2) ranged uniformly random vector, (5) Fasttext CharNgram based created vectors.
- Even though I am focusing on Turkish versions of the dataset, I believe "Preprocessor" can work for English dataset too. In future, I may add more language specific methods. 
- Due to laziness, I defined all my necessary arguments/configs/properties in "main.py". However, I also implemented argparse versions of the same properties (Sorry for hard-coded paths).
- I tested all training, evaluation, model/vocabulary saving/loading aspects of the code for several epochs without any problem (except out of memory errors =)).

## To-do 

- Run the current piece of code for the aforementioned datasets and define a text categorization baseline (for both Turkish and English).
- Attention and variational dropout will be added (to TextCNN).
- Different learning algorithms (DeepCNN, LSTM, GRU, any-kind-of-hybrid versions of those algorithms, transformers).
- CRF layer to be able to do NER experiments.
- For Turkish, I plan to add morphological disambiguation phase (https://github.com/erayyildiz/Neural-Morphological-Disambiguation-for-Turkish). 
- Different language models.

## How-to-run

**Important Note**: You need to change the torchtext backend to succesfully run this code, if you want to (1) work with Turkish Fasttext embeddings and (2) apply Fasttext-based OOV embedding generation. Changes can be found in "changes in torchtext.txt" file. 

After editing the hard coded paths in "main.py", it should not be a problem to start your own training process. 
If you succesfully train and save a model, you can evaluate the saved model interactively by changing the "run_mode" parameter from "train" to "evaluate_interactive". 
(I will edit here with more details later)

### References for code development: 
Below two repositories really helped me to write a decent and working code:
- https://github.com/bamtercelboo/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch
- https://github.com/bentrevett/pytorch-sentiment-analysis


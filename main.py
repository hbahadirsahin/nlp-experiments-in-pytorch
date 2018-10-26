from __future__ import print_function

import datetime
import os

import torch

from datahelper.dataset_reader import DatasetLoader
from datahelper.embedding_helper import OOVEmbeddingCreator
from datahelper.preprocessor import Preprocessor
from evaluation.evaluate import evaluate_interactive
from models.CNN import TextCnn
from training.train import train_iters
from utils.utils import save_vocabulary

dataset_properties = {"stop_word_path": "D:/nlpdata/stopwords/turkish",
                      # "stop_word_path": "D:/Anaconda3/nltk_data/corpora/stopwords/turkish",
                      "data_path": "D:/nlpdata/TWNERTC_TC_Coarse Grained NER_No_NoiseReduction.DUMP",
                      "embedding_vector": "fasttext.tr.300d",
                      "vector_cache": "D:/nlpdata/fasttext",
                      "pretrained_embedding_path": "D:/nlpdata/fasttext/wiki.tr",
                      "checkpoint_path": "",
                      "oov_embedding_type": "zeros",
                      "batch_size": 64
                      }

model_properties = {"use_pretrained_embed": True,
                    "embed_train_type": "nonstatic",
                    "use_padded_conv": True,
                    "keep_prob": 0.5,
                    "use_batch_norm": True,
                    "batch_norm_momentum": 0.1,
                    "batch_norm_affine": False,
                    "filter_count": 64,
                    "filter_sizes": [3, 4, 5],
                    # "run_mode": "eval_interactive",
                    "run_mode": "train",
                    }

training_properties = {"optimizer": "Adam",
                       "learning_rate": 0.05,
                       "weight_decay": 0,
                       "momentum": 0.9,
                       "norm_ratio": 10,
                       "epoch": 100,
                       "print_every_batch_step": 250,
                       "save_every_epoch": 1,
                       "topk": (5, 1),
                       "eval_every": 1,
                       }

evaluation_properties = {"model_path": "D:/PyTorchNLP/saved/2018-10-21/",
                         "sentence_vocab": "D:/PyTorchNLP/saved/vocab/sentence_vocab.dat",
                         "category_vocab": "D:/PyTorchNLP/saved/vocab/category_vocab.dat"
                         }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    assert model_properties["run_mode"] == "train" or \
           model_properties["run_mode"] == "eval_interactive"

    print("Initial device is", device)
    torch.backends.cudnn.benchmark = True

    stop_word_path = dataset_properties["stop_word_path"]
    data_path = dataset_properties["data_path"]
    vector_cache = dataset_properties["vector_cache"]
    fasttext_model_path = dataset_properties["pretrained_embedding_path"]

    oov_embedding_type = dataset_properties["oov_embedding_type"]
    batch_size = dataset_properties["batch_size"]

    embedding_vector = dataset_properties["embedding_vector"]

    save_dir = os.path.abspath(os.path.join(os.curdir, "saved", datetime.datetime.today().strftime('%Y-%m-%d')))
    save_dir_vocab = os.path.abspath(os.path.join(os.curdir, "saved", "vocab"))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(save_dir_vocab):
        os.makedirs(save_dir_vocab)
    print("Saving directory for models is", save_dir)
    print("Saving directory for vocabulary files is", save_dir_vocab)
    training_properties["save_path"] = save_dir

    print("Initialize Preprocessor")
    preprocessor = Preprocessor(stop_word_path,
                                is_remove_digit=True,
                                is_remove_punctuations=False)

    if model_properties["run_mode"] == "train":
        print("Initialize OOVEmbeddingCreator")
        unkembedding = OOVEmbeddingCreator(type=oov_embedding_type,
                                           fasttext_model_path=fasttext_model_path)

        print("Initialize DatasetLoader")
        datasetloader = DatasetLoader(data_path=data_path,
                                      vector=embedding_vector,
                                      preprocessor=preprocessor.preprocess,
                                      vector_cache=vector_cache,
                                      unk_init=unkembedding.create_oov_embedding)

        print("Loading train, validation and test sets")
        train, val, test = datasetloader.read_dataset(batch_size=batch_size)
        print("Loading vocabularies")
        sentence_vocab = datasetloader.sentence_vocab
        category_vocab = datasetloader.category_vocab
        print("Loading embeddings")
        pretrained_embeddings = datasetloader.sentence_vocab_vectors
        print("Updating properties")
        model_properties["device"] = device
        model_properties["vocab_size"] = pretrained_embeddings.size()[0]
        model_properties["embed_dim"] = pretrained_embeddings.size()[1]
        model_properties["num_class"] = len(category_vocab)
        model_properties["vocab"] = sentence_vocab
        model_properties["padding_id"] = sentence_vocab.stoi["<pad>"]
        model_properties["pretrained_weights"] = pretrained_embeddings

        print("Saving vocabulary files")
        save_vocabulary(sentence_vocab, os.path.abspath(os.path.join(save_dir_vocab, "sentence_vocab.dat")))
        save_vocabulary(category_vocab, os.path.abspath(os.path.join(save_dir_vocab, "category_vocab.dat")))
        print("Initialize model")
        model = TextCnn(model_properties).to(device)
        if dataset_properties["checkpoint_path"] is None or dataset_properties["checkpoint_path"] == "":
            print("Train process is starting from scratch!")
            train_iters(model=model,
                        train_iter=datasetloader.train_iter,
                        dev_iter=datasetloader.val_iter,
                        test_iter=datasetloader.test_iter,
                        device=device,
                        topk=training_properties["topk"],
                        training_properties=training_properties)
        else:
            checkpoint = torch.load(dataset_properties["checkpoint_path"])
            print("Train process is starting from checkpoint! Starting epoch is {}".format(checkpoint["epoch"]))
            train_iters(model=model,
                        train_iter=datasetloader.train_iter,
                        dev_iter=datasetloader.val_iter,
                        test_iter=datasetloader.test_iter,
                        device=device,
                        topk=training_properties["topk"],
                        training_properties=training_properties,
                        checkpoint=checkpoint)
    elif model_properties["run_mode"] == "eval_interactive":
        model_path = evaluation_properties["model_path"]
        sentence_vocab_path = evaluation_properties["sentence_vocab"]
        category_vocab_path = evaluation_properties["category_vocab"]

        print("Interactive evaluation mode for model {}:".format(model_path))

        evaluate_interactive(model_path=model_path,
                             sentence_vocab_path=sentence_vocab_path,
                             category_vocab_path=category_vocab_path,
                             preprocessor=preprocessor.preprocess,
                             topk=training_properties["topk"],
                             device=device)
    print("")

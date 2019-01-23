from __future__ import print_function

import logging.config
import random

from torchtext import data

from datahelper.preprocessor import Preprocessor
from embedding_helper import OOVEmbeddingCreator

SEED = 1234

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Dataset Loader")

class DatasetLoader(object):
    def __init__(self, data_path, vector, task, fix_length=0, min_freq=0, level="word", unk_init=None,
                 preprocessor=None,
                 vector_cache=None):
        assert data_path is not None and vector is not None

        self.data_path = data_path
        self.vector = vector
        self.preprocessor = preprocessor
        self.vector_cache = vector_cache
        self.unk_init = unk_init
        self.level = level
        self.task = task

        self.sentence_vocab = None
        self.category_vocab = None
        self.ner_vocab = None

        self.sentence_vocab_vectors = None

        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

        self.min_freq = min_freq
        self.fix_length = fix_length
        if min_freq <= 0:
            self.min_freq = 1
        if fix_length <= 0:
            self.fix_length = None

    '''
        This method is for character-level stuff.
        Since, torchtext do not let me do preprocess before tokenization in its normal flow (it always applies tokenization
        first, then does preprocess), I listened an advice from stackoverflow and wrote my own tokenizer which does 
        preprocess first, then applies tokenize preprocessed sentence into characters. Not happy with it, but it works =)
        :param sentence:
        :return: tokenized_sentence
    '''

    def evil_workaround_tokenizer(self, sentence):
        preprocessed_sentence = self.preprocessor(sentence)
        tokenized_sentence = list(preprocessed_sentence)
        return tokenized_sentence

    def create_fields(self, seq_input=True, seq_ner=True, seq_cat=False):
        if self.level == "word":
            sentence_field = data.Field(sequential=seq_input, preprocessing=self.preprocessor, pad_first=True,
                                        fix_length=self.fix_length)
        elif self.level == "char":
            sentence_field = data.Field(sequential=seq_input, tokenize=self.evil_workaround_tokenizer, fix_length=1014)
            # sentence_field = data.NestedField(nested_field)
        else:
            raise KeyError("Sentence_field is undefined!")

        ner_label_field = data.Field(sequential=seq_ner)
        category_label_field = data.LabelField(sequential=seq_cat)
        return sentence_field, ner_label_field, category_label_field

    def read_dataset(self, batch_size=128, split_ratio=0.7, format="tsv"):
        sf, nlf, clf = self.create_fields()
        if self.task == "classification":
            dataset = data.TabularDataset(path=self.data_path,
                                          format=format,
                                          skip_header=True,
                                          fields=[("category_labels", clf),
                                                  ("ner_labels", None),
                                                  ("sentence", sf)])
        elif self.task == "ner":
            dataset = data.TabularDataset(path=self.data_path,
                                          format=format,
                                          skip_header=True,
                                          fields=[("category_labels", None),
                                                  ("ner_labels", nlf),
                                                  ("sentence", sf)])
        else:
            raise ValueError("Training task is not defined! It can be 'classification' or 'ner'")

        logger.info("Splitting dataset into train/dev/test")
        train, val, test = self.create_splits(dataset, split_ratio)
        logger.info("Splitting done!")
        logger.info("Creating vocabulary")
        self.create_vocabs(train, sf, clf, nlf)
        logger.info("Vocabulary created!")
        logger.info("Creating iterators")
        self.create_iterator(train, val, test, batch_size)
        return train, val, test

    @staticmethod
    def create_splits(dataset, split_ratio):
        return dataset.split(split_ratio=split_ratio, random_state=random.seed(SEED))

    def create_vocabs(self, train, sentence_field, category_label_field, ner_label_field):
        if self.level == "word":
            sentence_field.build_vocab(train, vectors=self.vector, vectors_cache=self.vector_cache,
                                       unk_init=self.unk_init, min_freq=self.min_freq)
        else:
            sentence_field.build_vocab(train)

        self.sentence_vocab = sentence_field.vocab
        self.sentence_vocab_vectors = sentence_field.vocab.vectors

        if self.task == "classification":
            category_label_field.build_vocab(train)
            self.category_vocab = category_label_field.vocab
        else:
            ner_label_field.build_vocab(train)
            self.ner_vocab = ner_label_field.build_vocab


    def create_iterator(self, train, val, test, batch_size):
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(datasets=(train, val, test),
                                                                                    batch_sizes=(
                                                                                        batch_size, batch_size,
                                                                                        batch_size),
                                                                                    sort_key=lambda x: len(x.sentence),
                                                                                    repeat=False)


if __name__ == '__main__':
    stop_word_path = "D:/Anaconda3/nltk_data/corpora/stopwords/turkish"
    data_path = "D:/PyTorchNLP/data/turkish_test.DUMP"
    vector_cache = "D:/PyTorchNLP/data/fasttext"
    level = "char"
    is_char_level = True

    preprocessor = Preprocessor(stop_word_path,
                                is_remove_digit=False,
                                is_remove_punctuations=False,
                                is_char_level=is_char_level)

    unkembedding = OOVEmbeddingCreator(type="zeros",
                                       fasttext_model_path="D:/PyTorchNLP/data/fasttext/wiki.tr")

    dataset_helper = DatasetLoader(data_path=data_path,
                                   vector="fasttext.tr.300d",
                                   level=level,
                                   preprocessor=preprocessor.preprocess,
                                   vector_cache=vector_cache,
                                   unk_init=unkembedding.create_oov_embedding)

    print("Reading dataset")
    train, val, test = dataset_helper.read_dataset(batch_size=1)
    print(len(train), "-", len(val), "-", len(test))
    sentence_vocab = dataset_helper.sentence_vocab
    category_vocab = dataset_helper.category_vocab

    print("Vocab:", len(sentence_vocab))
    print("Vocab:", len(category_vocab))
    print("Most freq:", sentence_vocab.freqs.most_common(20))
    print("Most freq:", category_vocab.freqs.most_common(20))
    print("Itos:", sentence_vocab.itos[:50])
    print("Stoi:", category_vocab.stoi)

    train_iter = dataset_helper.train_iter
    val_iter = dataset_helper.val_iter
    test_iter = dataset_helper.test_iter

    print("Train iter size:", len(train_iter))
    print("Val iter size:", len(val_iter))
    print("Test iter size:", len(test_iter))

    for idx, batch in enumerate(train_iter):
        batch_x = batch.sentence
        print(batch_x.size())
        print(batch_x)
        # batch_x = torch.reshape(batch_x, (batch_x.size(0), batch_x.size(1)*batch_x.size(2)))
        if dataset_helper.level == "word":
            s = [sentence_vocab.itos[idx] for idx in batch_x]
        else:
            # s = [sentence_vocab.itos[char] for sentence in batch_x for word in sentence for char in word]
            s = [sentence_vocab.itos[idx] for idx in batch_x]

        print(idx, "-", s)
        print("")
        break

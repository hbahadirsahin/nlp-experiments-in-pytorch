from __future__ import print_function

import random

from torchtext import data

from datahelper.preprocessor import Preprocessor
from embedding_helper import OOVEmbeddingCreator

SEED = 1234


class DatasetLoader(object):
    def __init__(self, data_path, vector, unk_init=None, preprocessor=None, vector_cache=None):
        assert data_path is not None and vector is not None
        self.data_path = data_path
        self.vector = vector
        self.preprocessor = preprocessor
        self.vector_cache = vector_cache
        self.unk_init = unk_init

        self.sentence_vocab = None
        self.category_vocab = None
        self.ner_vocab = None

        self.sentence_vocab_vectors = None

        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def create_fields(self, seq_input=True, seq_ner=True, seq_cat=False, fix_length=None):
        sentence_field = data.Field(sequential=seq_input, preprocessing=self.preprocessor, pad_first=True,
                                    fix_length=fix_length)
        ner_label_field = data.Field(sequential=seq_ner)
        category_label_field = data.LabelField(sequential=seq_cat)
        return sentence_field, ner_label_field, category_label_field

    def read_dataset(self, batch_size=128, split_ratio=0.7, format="tsv"):
        sf, nlf, clf = self.create_fields()
        dataset = data.TabularDataset(path=self.data_path,
                                      format=format,
                                      skip_header=True,
                                      fields=[("category_labels", clf),
                                              ("ner_labels", None),
                                              ("sentence", sf)])
        print("Splitting dataset into train/dev/test")
        train, val, test = self.create_splits(dataset, split_ratio)
        print("Splitting done!")
        print("Creating vocabulary")
        self.create_vocabs(train, sf, clf)
        print("Vocabulary created!")
        print("Creating iterators")
        self.create_iterator(train, val, test, batch_size)
        return train, val, test

    def read_dataset_for_test(self, batch_size=128, format="tsv"):
        sf, _, clf = self.create_fields()
        test_dataset = data.TabularDataset(path=self.data_path,
                                           format=format,
                                           skip_header=True,
                                           fields=[("category_labels", clf),
                                                   ("ner_labels", None),
                                                   ("sentence", sf)])
        return data.BucketIterator.splits(datasets=test_dataset,
                                          batch_size=batch_size,
                                          sort_key=lambda x: len(x.sentence),
                                          repeat=False)

    @staticmethod
    def create_splits(dataset, split_ratio):
        return dataset.split(split_ratio=split_ratio, random_state=random.seed(SEED))

    def create_vocabs(self, train, sentence_field, category_label_field, min_freq=1):
        sentence_field.build_vocab(train, vectors=self.vector, vectors_cache=self.vector_cache, unk_init=self.unk_init,
                                   min_freq=min_freq)
        category_label_field.build_vocab(train)

        self.sentence_vocab = sentence_field.vocab
        self.category_vocab = category_label_field.vocab

        self.sentence_vocab_vectors = sentence_field.vocab.vectors

    def create_iterator(self, train, val, test, batch_size):
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(datasets=(train, val, test),
                                                                                    batch_sizes=(
                                                                                        batch_size, batch_size,
                                                                                        batch_size),
                                                                                    sort_key=lambda x: len(x.sentence),
                                                                                    repeat=False)


if __name__ == '__main__':
    stop_word_path = "D:/Anaconda3/nltk_data/corpora/stopwords/turkish"
    data_path = "D:/PyTorchNLP/data/twnertc_basic_tr.DUMP"
    vector_cache = "D:/PyTorchNLP/data/fasttext"

    preprocessor = Preprocessor(stop_word_path,
                                is_remove_digit=False,
                                is_remove_punctuations=False)

    unkembedding = OOVEmbeddingCreator(type="zeros",
                                       fasttext_model_path="D:/PyTorchNLP/data/fasttext/wiki.tr")

    dataset_helper = DatasetLoader(data_path=data_path,
                                   vector="fasttext.tr.300d",
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
    print("Itos:", sentence_vocab.itos[:10])
    print("Stoi:", category_vocab.stoi)

    train_iter = dataset_helper.train_iter
    val_iter = dataset_helper.val_iter
    test_iter = dataset_helper.test_iter

    print("Train iter size:", len(train_iter))
    print("Val iter size:", len(val_iter))
    print("Test iter size:", len(test_iter))

    for idx, batch in enumerate(train_iter):
        batch_x = batch.sentence
        s = [sentence_vocab.itos[idx] for idx in batch_x]
        print(idx, "-", s)
        print("")

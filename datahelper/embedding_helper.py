import logging.config

import torch
from gensim.models import FastText

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Embedding Helper")

class OOVEmbeddingCreator(object):
    def __init__(self, type="zeros", range=(-0.25, 0.25), fasttext_model_path="None"):
        assert type == "zeros" or type == "ones" or type == "random" or type == "uniform" or type == "fasttext_oov"
        self.type = type
        self.range = range
        self.fasttext_model_path = fasttext_model_path
        self.random_emb = None
        self.uniform_emb = None
        logger.info("> OOV Embedding mode: %s", self.type)
        if self.type == "fasttext_oov":
            assert self.fasttext_model_path is not None
            logger.info(">> Fasttext model will be loaded and embeddings for OOV words will be calculated by using it!")
            logger.info(">> Beware that the process may take a while due to this process!")
            self.model = FastText.load_fasttext_format(self.fasttext_model_path)

    def create_oov_embedding(self, vector, word=None):
        if self.type == "zeros":
            return torch.zeros(vector.size())
        elif self.type == "ones":
            return torch.ones(vector.size())
        elif self.type == "random":
            if self.random_emb is None:
                self.random_emb = torch.randn(vector.size())
            return self.random_emb
        elif self.type == "uniform":
            if self.uniform_emb is None:
                self.uniform_emb = torch.FloatTensor(vector.size()).uniform_(self.range[0], self.range[1])
            return self.uniform_emb
        elif self.type == "fasttext_oov":
            try:
                res = torch.from_numpy(self.model.wv.word_vec(word))
            except KeyError:
                res = torch.randn(vector.size())
            return res

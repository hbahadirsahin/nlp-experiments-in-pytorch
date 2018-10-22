import re
import string

all_letters = string.ascii_letters + ".,;"


class Preprocessor(object):
    def __init__(self, stop_word_path, is_remove_digit=True, is_remove_punctuations=True):
        self.stop_words = self.load_stop_words(stop_word_path)
        self.is_remove_digit = is_remove_digit
        self.is_remove_punctuations = is_remove_punctuations

    @staticmethod
    def load_stop_words(path):
        return set(line.strip() for line in open(path))

    @staticmethod
    def remove_line_breaks(sentence):
        return sentence.replace("\r", "").replace("\n", "")

    @staticmethod
    def remove_punctuations(sentence):
        return "".join([ch for ch in sentence if ch not in string.punctuation])

    @staticmethod
    def remove_multiple_white_spaces(sentence):
        return " ".join(sentence.split())

    def remove_stop_words(self, sentence):
        return " ".join([word for word in sentence.split() if word not in self.stop_words])

    @staticmethod
    def to_lowercase(sentence):
        return sentence.lower()

    @staticmethod
    def remove_digits(sentence):
        return "".join([word for word in sentence if not word.isdigit()])

    @staticmethod
    def replace_digits(sentence):
        return re.sub("\d+", "<NUM>", sentence)

    @staticmethod
    def remove_alphanumeric(sentence):
        return "".join([word for word in sentence if not word.isalnum()])

    @staticmethod
    def remove_non_utf8(sentence):
        return bytes(sentence, "utf-8").decode("utf-8", "ignore")

    @staticmethod
    def change_currency_characters(sentence):
        return sentence.replace('$', 'dolar').replace('£', 'sterlin').replace('€', 'euro')

    def preprocess(self, sentence):
        # TorchText returns a list of words instead of a normal sentence.
        # First, create the sentence again. Then, do preprocess. Finally, return the preprocessed sentence as list
        # of words
        x = " ".join(sentence)
        x = self.to_lowercase(x)
        x = self.change_currency_characters(x)

        if self.is_remove_punctuations:
            x = self.remove_punctuations(x)

        x = self.remove_stop_words(x)

        if self.is_remove_digit:
            x = self.remove_digits(x)
        else:
            x = self.replace_digits(x)

        x = self.remove_line_breaks(x)
        x = self.remove_multiple_white_spaces(x)
        return (x.strip()).split()


if __name__ == '__main__':
    stop_word_path = "D:/Anaconda3/nltk_data/corpora/stopwords/turkish"
    dataset_path = "D:/PyTorchNLP/data/twnertc_basic_tr.DUMP"

    preprocessor = Preprocessor(stop_word_path,
                                is_remove_digit=False,
                                is_remove_punctuations=False)

    with open(dataset_path, encoding="utf-8") as dataset:
        for counter, line in enumerate(dataset):
            line_tokens = line.split('\t')
            sentence = line_tokens[2]
            print(" > ", sentence)
            print(" = ", preprocessor.preprocess(sentence))
            if counter == 10:
                break

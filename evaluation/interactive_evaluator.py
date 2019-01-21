import logging.config

import pkg_resources

try:
    pkg_resources.get_distribution("spacy")
except pkg_resources.DistributionNotFound:
    print("Spacy has not been found! As sentence tokenizer .split() will be used!")
    HAS_SPACY = False
else:
    import spacy

    HAS_SPACY = True
import torch
import torch.nn.functional as F

from utils.utils import load_best_model, load_vocabulary

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Evaluator")


class InteractiveEvaluator(object):
    def __init__(self, device="cpu"):
        self.device = device

    def evaluate_interactive(self, model_path, sentence_vocab_path, category_vocab_path, preprocessor, topk):
        sentence_vocab = load_vocabulary(sentence_vocab_path)
        category_vocab = load_vocabulary(category_vocab_path)

        model = load_best_model(model_path)
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            while True:
                try:
                    sentence = input("Enter a test sentence (Type q or quit to exit!):")
                except ValueError:
                    logger.error("Invalid input. Try again! (Type q or quit to exit!)")
                    continue

                if sentence.lower() != "q" and sentence.lower() != "quit":
                    # Below 5 lines of weird looking code is for tokenizing a test input, correctly.
                    # Obviously, sentence.split() does not work if the sentence has punctuations to tokenize.
                    # Example: "a, b c." sentence should be tokenized as "a , b c . ".
                    # That's why I added spacy tokenizer. And, lucky me it works for Turkish, too =)
                    # Note that I added this tokenization to preprocessor; however, it takes too much time to prepare a
                    # whole dataset in training process. Since the dataset I am using is already tokenized as it should be,
                    # I wrote the below code to only evaluation process which is less shorter than my comment to explain
                    # this situation =)
                    if HAS_SPACY:
                        nlp_tokenizer = spacy.load("en")
                        doc = nlp_tokenizer(sentence.lower())
                        tokenized_sentence = [token.text for token in doc]
                        preprocessed_sentence = preprocessor(tokenized_sentence)
                        temp = nlp_tokenizer(" ".join(preprocessed_sentence))
                        preprocessed_sentence = [token.text for token in temp]
                    else:
                        preprocessed_sentence = preprocessor(sentence.lower().split())

                    indexed_test_sentence = [sentence_vocab.stoi[token] for token in preprocessed_sentence]

                    tensored_test_sentence = torch.LongTensor(indexed_test_sentence).to(self.device)

                    tensored_test_sentence = tensored_test_sentence.unsqueeze(1)

                    logit = model(tensored_test_sentence)
                    probs = F.softmax(logit, dim=1)

                    predicted_category_probs, predicted_category_ids = probs.topk(topk, 1, True, True)

                    predicted_category_ids = predicted_category_ids.t()

                    predicted_labels = []
                    for idx in predicted_category_ids:
                        predicted_labels.append(category_vocab.itos[idx])

                    if topk == 1:
                        logger.info("Predicted category is {} with probability {}".format(predicted_labels[0],
                                                                                          predicted_category_probs[0][
                                                                                              0].item()))
                    else:
                        logger.info("Top-{} predicted labels are as follows in order:".format(topk))
                        for idx, label in enumerate(predicted_labels):
                            logger.info("> {} - Predicted category is {} with probability {:.4f}".format(idx + 1,
                                                                                                         label,
                                                                                                         predicted_category_probs[
                                                                                                             0][
                                                                                                             idx].item()))
                else:
                    logger.info("Interactive evaluation ends!")
                    break

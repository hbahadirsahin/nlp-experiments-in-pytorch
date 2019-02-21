import logging.config

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("NerScorer")

class NerScorer(object):
    def __init__(self, ner_vocab):
        super(NerScorer, self).__init__()
        self.ner_vocab = ner_vocab
        self.token_accuracy = 0
        self.avg_macro_precision = 0
        self.avg_macro_recall = 0
        self.avg_macro_f1 = 0
        self.macro_precision = {}
        self.macro_recall = {}
        self.macro_f1 = {}

    def token_level_accuracy(self, prediction, ground_truth):
        token_count = 0
        matched = 0

        for p_seq, gt_seq in zip(prediction, ground_truth):
            for p, gt in zip(p_seq, gt_seq):
                token_count += 1
                if p == gt:
                    matched += 1

        self.token_accuracy = matched * 100.0 / token_count

    def __initialize_dict(self):
        d = {}
        for v in self.ner_vocab.stoi:
            d[self.ner_vocab.stoi[v]] = 0
        return d

    def __add_to_dict(self, d, tag):
        if tag in d:
            d[tag] += 1
        else:
            d[tag] = 1
        return d

    def __calculate_tag_f1(self, f1, tp, fp, fn):
        precision = {}
        recall = {}
        for tag in tp:
            precision[tag] = tp[tag] / (tp[tag] + fp[tag] + 1e-16)
            recall[tag] = tp[tag] / (tp[tag] + fn[tag] + 1e-16)
            f1[tag] = (2 * precision[tag] * recall[tag] / (precision[tag] + recall[tag] + 1e-16)) * 100
        self.macro_f1 = f1
        self.macro_precision = precision
        self.macro_recall = recall

    def __calculate_mean_f1(self):
        self.avg_macro_f1 = sum(self.macro_f1.values()) / float(len(self.macro_f1))
        self.avg_macro_precision = sum(self.macro_precision.values()) / float(len(self.macro_precision))
        self.avg_macro_recall = sum(self.macro_recall.values()) / float(len(self.macro_recall))

    def f1_score(self, prediction, ground_truth):
        true_positives = self.__initialize_dict()
        false_positives = self.__initialize_dict()
        false_negatives = self.__initialize_dict()
        f1 = self.__initialize_dict()

        for p_seq, gt_seq in zip(prediction, ground_truth):
            for p, gt in zip(p_seq, gt_seq):
                if p == gt:
                    true_positives = self.__add_to_dict(true_positives, gt)
                else:
                    false_negatives = self.__add_to_dict(false_negatives, gt)
                    false_positives = self.__add_to_dict(false_positives, p)
        self.__calculate_tag_f1(f1, true_positives, false_positives, false_negatives)
        self.__calculate_mean_f1()

    def print_detailed_score_log(self):
        logger.info("--------------------")
        logger.info("Detailed Tag-Based Score")
        for tag in self.macro_f1:
            logger.info("Tag: {} - Precision: {:.4f} - Recall: {:.4f} - F1: {:.4f}".format(self.ner_vocab.itos[tag],
                                                                                           self.macro_precision[tag],
                                                                                           self.macro_recall[tag],
                                                                                           self.macro_f1[tag]))
        logger.info("--------------------")

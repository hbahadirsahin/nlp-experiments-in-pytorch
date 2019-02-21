import logging.config

import torch

from models.GRU import GRU
from models.LSTM import LSTMBase
from utils.utils import load_best_model

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Evaluator")


class SingleModelNerEvaluator(object):
    def __init__(self, device, is_vali):
        self.device = device
        self.is_vali = is_vali

    def evaluate_iter(self, model, input, save_path, scorer, detailed_ner_log=True):
        total_loss = 0
        macro_f1 = 0
        macro_precision = 0
        macro_recall = 0
        total_token_acc = 0

        if not self.is_vali:
            logger.info("Test mode!")
            model = load_best_model(save_path)
        else:
            logger.info("Validation mode!")
        model.eval()

        full_ground_truth_list = list()
        full_prediction_list = list()

        with torch.no_grad():
            for batch in input:
                batch_x = batch.sentence.to(self.device)
                batch_y = batch.ner_labels.to(self.device)

                if isinstance(model, GRU) or isinstance(model, LSTMBase):
                    model.hidden = model.init_hidden(batch_x.size(1))

                pred_scores, predictions = model.decode(batch_x)

                batch_y = batch_y.permute(1, 0)

                scorer.token_level_accuracy(predictions, batch_y)

                full_ground_truth_list.extend(batch_y.tolist())
                full_prediction_list.extend(predictions)

                token_level_accuracy = scorer.token_accuracy

                total_token_acc += token_level_accuracy

                torch.cuda.empty_cache()

            scorer.f1_score(full_prediction_list, full_ground_truth_list)
            macro_f1 = scorer.avg_macro_f1
            macro_precision = scorer.avg_macro_precision
            macro_recall = scorer.avg_macro_recall
            current_token_acc = total_token_acc / len(input)

            if detailed_ner_log:
                scorer.print_detailed_score_log()

            return macro_f1, macro_precision, macro_recall, current_token_acc

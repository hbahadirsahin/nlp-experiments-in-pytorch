import logging.config

import torch

from models.GRU import GRU
from models.LSTM import LSTMBase
from utils.utils import calculate_accuracy, calculate_topk_accuracy, load_best_model

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Evaluator")


class SingleModelNerEvaluator(object):
    def __init__(self, device, is_vali):
        self.device = device
        self.is_vali = is_vali

    def evaluate_iter(self, model, input, save_path):
        total_loss = 0
        total_f1 = 0

        if not self.is_vali:
            logger.info("Test mode!")
            model = load_best_model(save_path)
        else:
            logger.info("Validation mode!")
        model.eval()

        with torch.no_grad():
            for batch in input:
                batch_x = batch.sentence.to(self.device)
                batch_y = batch.ner_labels.to(self.device)

                if isinstance(model, GRU) or isinstance(model, LSTMBase):
                    model.hidden = model.init_hidden(batch_x.size(1))

                pred_scores, predictions = model.decode(batch_x)

                # total_loss += loss
                print(predictions)

                torch.cuda.empty_cache()

            current_loss = total_loss / len(input)
            current_f1 = total_f1 / len(input)

            return current_loss, current_f1

import logging.config
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from custom_optimizer import OpenAIAdam, NoamOptimizer, Padam
from evaluation.evaluator import Evaluator
from models.GRU import GRU
from models.LSTM import LSTMBase
from training.single_model_trainer import SingleModelTrainer
from utils.utils import time_since, save_best_model
from scorer.ner_scorer import NerScorer

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Trainer")


class SingleModelNerTrainer(SingleModelTrainer):
    def __init__(self, training_properties, train_iter, dev_iter, test_iter, device):
        super(SingleModelNerTrainer, self).__init__(training_properties, train_iter, dev_iter, test_iter, device)

        self.scorer = NerScorer()
        self.dev_evaluator, self.test_evaluator = Evaluator().evaluator_factory("single_model_ner_evaluator",
                                                                                self.device)

    def train_iters(self, model, checkpoint=None):
        optimizer = self.init_optimizer(model)

        start = time.time()
        old_path = None
        best_vali_f1 = -1
        best_vali_token_acc = -1
        start_epoch = 1

        if checkpoint is not None:
            model.load(checkpoint["model"])
            if self.optimizer_type == "Noam":
                optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_vali_f1 = checkpoint["best_vali_f1"]
            best_vali_token_acc = checkpoint["best_vali_token_acc"]

        del checkpoint
        torch.cuda.empty_cache()

        logger.info("Training...")
        for e in range(start_epoch, self.epoch + 1):
            total_loss, train_f1 = self.train(model=model,
                                              optimizer=optimizer,
                                              scheduler=None)

            self.print_epoch(start, e, total_loss, train_f1)

            if e % self.eval_every == 0:
                vali_f1, vali_token_acc = self.dev_evaluator.evaluate_iter(model=model,
                                                                           input=self.dev_iter,
                                                                           save_path=self.save_path,
                                                                           scorer=self.scorer)
                if best_vali_f1 < vali_f1:
                    best_vali_token_acc = vali_token_acc
                    best_vali_f1 = vali_f1
                    save_best_model(model, self.save_path)

                self.print_validation(vali_token_acc, best_vali_token_acc, vali_f1, best_vali_f1)

            if e % self.save_every == 0:
                filename = "saved_model_step{}.pt".format(e)
                out_path = os.path.abspath(os.path.join(self.save_path, filename))
                if old_path is not None:
                    os.remove(old_path)
                if self.optimizer_type == "Noam":
                    torch.save({
                        "epoch": e,
                        "best_vali_f1": best_vali_f1,
                        "best_vali_token_acc": best_vali_token_acc,
                        'model': model,
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, out_path)
                else:
                    torch.save({
                        "epoch": e,
                        "best_vali_f1": best_vali_f1,
                        "best_vali_token_acc": best_vali_token_acc,
                        'model': model,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, out_path)
                old_path = out_path

        test_f1, test_token_acc = self.test_evaluator.evaluate_iter(model=model,
                                                                    input=self.test_iter,
                                                                    save_path=self.save_path,
                                                                    scorer=self.scorer)

        self.print_test(test_token_acc, test_f1)

    def train(self, model, optimizer, scheduler):
        epoch_loss = 0
        epoch_total_f1 = 0
        step = 1
        model.train()

        for batch in self.train_iter:
            if self.optimizer_type == "Noam":
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            batch_x = batch.sentence.to(self.device)
            batch_y = batch.ner_labels.to(self.device, non_blocking=True)

            if isinstance(model, GRU) or isinstance(model, LSTMBase):
                model.hidden = model.init_hidden(batch_x.size(1))

            try:
                loss, kl_loss = model(batch_x, batch_y)
                # f1_score = calculate_accuracy(predictions, batch_y)

                loss.backward()

                if 0.0 < self.norm_ratio:
                    nn.utils.clip_grad_norm_(model.parameters(), self.norm_ratio)

                if self.optimizer_type == "Noam":
                    optimizer.optimizer.step()
                else:
                    optimizer.step()

                if scheduler is not None and step % 500 == 0:
                    scheduler.step(step)

                step += 1

                epoch_loss += loss.item()
                # epoch_total_f1 += f1_score

                if step % self.print_every == 0:
                    self.print_step(step, loss.item(), 0)

                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warning('Ran out of memory, skipping batch %d', step)
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                else:
                    raise e

        return epoch_loss / len(self.train_iter), epoch_total_f1 / len(self.train_iter)

    def print_step(self, step, loss, f1_score):
        logger.info("Batch {}/{} - "
                    "Batch Loss: {:.4f} - "
                    "Batch F1: {:.4f}".format(step,
                                              len(self.train_iter),
                                              loss,
                                              f1_score))

    def print_epoch(self, start, e, total_loss, train_f1):
        logger.info("{} - "
                    "Epoch {}/{} - "
                    "Loss: {:.4f} - "
                    "F1-Score: {:.4f}".format(time_since(start, e / self.epoch),
                                              e,
                                              self.epoch,
                                              total_loss,
                                              train_f1))

    def print_validation(self, vali_f1, best_vali_f1, vali_token_acc, best_vali_token_acc):
        logger.info("Validation F1: {:.4f} (Best: {:.4f}) - "
                    "Validation Token Level Accuracy: {:.4f} (Best: {:.4f}) - ".format(vali_token_acc,
                                                                                       best_vali_token_acc,
                                                                                       vali_f1,
                                                                                       best_vali_f1))

    def print_test(self, test_token_acc, test_f1):
        logger.info("Test F1: {:.4f} - "
                    "Test Token Level Accuracy: {:.4f} - ".format(test_f1,
                                                                  test_token_acc))


import logging.config
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from adabound import adabound

from custom_optimizer import OpenAIAdam, NoamOptimizer, Padam
from evaluation.evaluator import Evaluator
from models.GRU import GRU
from models.LSTM import LSTMBase
from utils.utils import time_since, save_best_model
from scorer.accuracy_scorer import AccuracyScorer

logging.config.fileConfig(fname='./config/config.logger', disable_existing_loggers=False)
logger = logging.getLogger("Trainer")


class SingleModelTrainer(object):
    def __init__(self, training_properties, datasetloader, device):
        self.task = training_properties["task"]
        self.optimizer_type = training_properties["optimizer"]
        self.learning_rate = training_properties["learning_rate"]
        self.weight_decay = training_properties["weight_decay"]
        self.momentum = training_properties["momentum"]
        self.norm_ratio = training_properties["norm_ratio"]
        self.epoch = training_properties["epoch"]
        self.topk = training_properties["topk"]
        self.print_every = training_properties["print_every_batch_step"]
        self.save_every = training_properties["save_every_epoch"]
        self.eval_every = training_properties["eval_every"]
        self.save_path = training_properties["save_path"]

        self.openAIAdamSchedulerType = training_properties["scheduler_type"]
        self.amsgrad = training_properties["amsgrad"]
        self.partial_adam = training_properties["partial_adam"]

        self.final_learning_rate = training_properties["final_learning_rate"]
        self.amsbound = training_properties["amsbound"]

        self.train_iter = datasetloader.train_iter
        self.dev_iter = datasetloader.val_iter
        self.test_iter = datasetloader.test_iter

        self.device = device

        self.scorer = AccuracyScorer(self.topk)
        self.dev_evaluator, self.test_evaluator = Evaluator().evaluator_factory("single_model_evaluator", self.device)

    def init_optimizer(self, model):
        logger.info("Optimizer type is %s!", self.optimizer_type)

        if self.optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                             momentum=self.momentum)
        elif self.optimizer_type == "OpenAIAdam":
            return OpenAIAdam(model.parameters(), lr=self.learning_rate, schedule=self.openAIAdamSchedulerType,
                              warmup=0.002, t_total=len(self.train_iter) * self.epoch)
        elif self.optimizer_type == "Noam":
            return NoamOptimizer(300, 1, 400,
                                 optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        elif self.optimizer_type == "Padam":
            return Padam(model.parameters(), lr=self.learning_rate, amsgrad=self.amsgrad, partial=self.partial_adam,
                         weight_decay=self.weight_decay)
        elif self.optimizer_type == "adabound":
            return adabound(model.parameters(), lr=self.learning_rate, final_lr=self.final_learning_rate,
                            weight_decay=self.weight_decay, amsbound=self.amsbound)
        else:
            raise ValueError("Invalid optimizer type! Choose Adam, SGD, Padam, NoamOptimizer or OpenAIAdam!")

    def train_iters(self, model, checkpoint=None):
        optimizer = self.init_optimizer(model)

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        if isinstance(model, GRU):
            criterion = nn.NLLLoss().to(self.device)
        else:
            criterion = nn.CrossEntropyLoss().to(self.device)

        start = time.time()
        old_path = None
        best_vali_acc = -1
        best_vali_loss = -1
        best_vali_acc_topk = -1
        start_epoch = 1

        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer_type == "Noam":
                optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_vali_acc = checkpoint["best_vali_acc"]
            best_vali_loss = checkpoint["best_vali_loss"]
            best_vali_acc_topk = checkpoint["best_vali_acc_topk"]

        # Memory problems of PyTorch is giving me headaches...
        del checkpoint
        torch.cuda.empty_cache()

        logger.info("Training...")
        for e in range(start_epoch, self.epoch + 1):
            total_loss, cross_entropy_loss, kl_loss, accuracy, accuracy_topk = self.train(model=model,
                                                                                          optimizer=optimizer,
                                                                                          scheduler=None,
                                                                                          criterion=criterion)

            self.print_epoch(start, e, cross_entropy_loss, kl_loss, total_loss, accuracy, accuracy_topk)

            if e % self.eval_every == 0:
                vali_loss, vali_accuracy, vali_accuracy_topk = self.dev_evaluator.evaluate_iter(model=model,
                                                                                                input=self.dev_iter,
                                                                                                criterion=criterion,
                                                                                                save_path=self.save_path,
                                                                                                scorer=self.scorer)
                if best_vali_acc < vali_accuracy:
                    best_vali_loss = vali_loss
                    best_vali_acc = vali_accuracy
                    best_vali_acc_topk = vali_accuracy_topk
                    save_best_model(model, self.save_path)

                self.print_validation(vali_loss, best_vali_loss, vali_accuracy, best_vali_acc, vali_accuracy_topk,
                                      best_vali_acc_topk)

            if e % self.save_every == 0:
                filename = "saved_model_step{}.pt".format(e)
                out_path = os.path.abspath(os.path.join(self.save_path, filename))
                if old_path is not None:
                    os.remove(old_path)
                if self.optimizer_type == "Noam":
                    torch.save({
                        "epoch": e,
                        "best_vali_acc": best_vali_acc,
                        "best_vali_loss": best_vali_loss,
                        "best_vali_acc_topk": best_vali_acc_topk,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, out_path)
                else:
                    torch.save({
                        "epoch": e,
                        "best_vali_acc": best_vali_acc,
                        "best_vali_loss": best_vali_loss,
                        "best_vali_acc_topk": best_vali_acc_topk,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, out_path)

                old_path = out_path

        test_loss, test_accuracy, test_accuracy_topk = self.test_evaluator.evaluate_iter(model=model,
                                                                                         input=self.test_iter,
                                                                                         criterion=criterion,
                                                                                         save_path=self.save_path,
                                                                                         scorer=self.scorer)

        self.print_test(test_loss, test_accuracy, test_accuracy_topk)

    def train(self, model, optimizer, scheduler, criterion):
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_total_acc = 0
        epoch_total_acc_topk = 0
        step = 1
        model.train()

        for batch in self.train_iter:
            if self.optimizer_type == "Noam":
                optimizer.optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            batch_x = batch.sentence.to(self.device)
            batch_y = batch.category_labels.to(self.device, non_blocking=True)

            if isinstance(model, GRU) or isinstance(model, LSTMBase):
                model.hidden = model.init_hidden(batch_x.size(1))

            try:
                predictions, kl_loss = model(batch_x)

                loss = criterion(predictions, batch_y)
                accuracy = self.scorer.calculate_accuracy(predictions, batch_y)
                accuracy_topk = self.scorer.calculate_topk_accuracy(predictions, batch_y)

                total_loss = loss + kl_loss / 10

                total_loss.backward()

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
                epoch_kl_loss += kl_loss.item()
                epoch_total_acc += accuracy
                epoch_total_acc_topk += accuracy_topk[0].item()

                if step % self.print_every == 0:
                    self.print_step(step, loss.item(), kl_loss.item(), accuracy, accuracy_topk[0].item())

                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warning('Ran out of memory, skipping batch %d', step)
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                else:
                    raise e

        epoch_total_loss = epoch_loss + epoch_kl_loss
        return epoch_total_loss / len(self.train_iter), epoch_loss / len(self.train_iter), epoch_kl_loss / len(
            self.train_iter), epoch_total_acc / len(self.train_iter), epoch_total_acc_topk / len(self.train_iter)

    def print_step(self, step, loss, kl_loss, accuracy, accuracy_topk):
        logger.info("Batch {}/{} - "
                    "Batch Loss: {:.4f} - "
                    "Batch KL Loss: {:.4f} - "
                    "Batch Accuracy: {:.4f} - "
                    "Batch Accuracy Top-{} {:.4f}".format(step,
                                                          len(self.train_iter),
                                                          loss,
                                                          kl_loss,
                                                          accuracy,
                                                          self.topk[0],
                                                          accuracy_topk))

    def print_epoch(self, start, e, cross_entropy_loss, kl_loss, total_loss, accuracy, accuracy_topk):
        logger.info("{} - "
                    "Epoch {}/{} - "
                    "Cross Entropy Loss: {:.4f} - "
                    "KL Loss: {:.4f} - "
                    "Loss: {:.4f} - "
                    "Accuracy: {:.4f} - "
                    "Accuracy Top-{}: {:.4f}".format(time_since(start, e / self.epoch),
                                                     e,
                                                     self.epoch,
                                                     cross_entropy_loss,
                                                     kl_loss,
                                                     total_loss,
                                                     accuracy,
                                                     self.topk[0],
                                                     accuracy_topk))

    def print_validation(self, vali_loss, best_vali_loss, vali_accuracy, best_vali_acc, vali_accuracy_topk,
                         best_vali_acc_topk):
        logger.info("Validation Loss: {:.4f} (Best: {:.4f}) - "
                    "Validation Accuracy: {:.4f} (Best: {:.4f}) - "
                    "Validation Accuracy Top-{}: {:.4f} (Best: {:.4f})".format(vali_loss,
                                                                               best_vali_loss,
                                                                               vali_accuracy,
                                                                               best_vali_acc,
                                                                               self.topk[0],
                                                                               vali_accuracy_topk,
                                                                               best_vali_acc_topk))

    def print_test(self, test_loss, test_accuracy, test_accuracy_topk):
        logger.info("Test Loss: {:.4f} - "
                    "Test Accuracy: {:.4f} - "
                    "Test Accuracy Top-{}: {:.4f}".format(test_loss,
                                                          test_accuracy,
                                                          self.topk[0],
                                                          test_accuracy_topk))

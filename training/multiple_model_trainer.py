import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from custom_optimizer import OpenAIAdam
from evaluation.evaluator import Evaluator
from utils.utils import time_since, calculate_accuracy, save_best_model, calculate_topk_accuracy, \
    scheduled_annealing_strategy


class MultipleModelTrainer(object):
    def __init__(self, training_properties, train_iter, dev_iter, test_iter, device):
        self.optimizer_type = training_properties["optimizer"]
        self.learning_rate = training_properties["learning_rate"]
        self.weight_decay = training_properties["weight_decay"]
        self.momentum = training_properties["momentum"]
        self.epoch = training_properties["epoch"]
        self.topk = training_properties["topk"]
        self.print_every = training_properties["print_every_batch_step"]
        self.save_every = training_properties["save_every_epoch"]
        self.eval_every = training_properties["eval_every"]
        self.save_path = training_properties["save_path"]

        self.openAIAdamSchedulerType = training_properties["scheduler_type"]

        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter

        self.device = device

        self.dev_evaluator, self.test_evaluator = Evaluator().evaluator_factory("multiple_model_evaluator", self.device)

    def init_optimizer(self, model):
        print("Optimizer type is {} !".format(self.optimizer_type))

        if self.optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                             momentum=self.momentum)
        elif self.optimizer_type == "OpenAIAdam":
            return OpenAIAdam(model.parameters(), lr=self.learning_rate, schedule=self.openAIAdamSchedulerType,
                              warmup=0.002, t_total=len(self.train_iter) * self.epoch)
        else:
            raise ValueError("Invalid optimizer type! Choose Adam or SGD!")

    def train_iters_multi_model(self, models, checkpoint=None):
        # Under the assumption of models is a list that contains encoder, decoder and classifier in order.
        encoder = models[0]
        decoder = models[1]
        classifier = models[2]

        encoder_optimizer = self.init_optimizer(encoder)
        decoder_optimizer = self.init_optimizer(decoder)
        classifier_optimizer = self.init_optimizer(classifier)

        reconstruction_criterion = nn.CrossEntropyLoss().to(self.device)
        supervised_criterion = nn.NLLLoss().to(self.device)

        start = time.time()
        old_path = None
        best_vali_acc = -1
        best_vali_loss = -1
        best_vali_acc_topk = -1
        start_epoch = 1

        if checkpoint is not None:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            classifier.load_state_dict(checkpoint["classifier_stat_dict"])
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
            classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_vali_acc = checkpoint["best_vali_acc"]
            best_vali_loss = checkpoint["best_vali_loss"]
            best_vali_acc_topk = checkpoint["best_vali_acc_topk"]

        print("Training...")
        for e in range(start_epoch, self.epoch + 1):
            alpha = scheduled_annealing_strategy(epoch=e, max_epoch=self.epoch)
            total_loss, reconst_loss, supervised_loss, accuracy, accuracy_topk = self.train(encoder=encoder,
                                                                                            decoder=decoder,
                                                                                            classifier=classifier,
                                                                                            encoder_optimizer=encoder_optimizer,
                                                                                            decoder_optimizer=decoder_optimizer,
                                                                                            classifier_optimizer=classifier_optimizer,
                                                                                            reconst_criterion=reconstruction_criterion,
                                                                                            supervised_criterion=supervised_criterion,
                                                                                            alpha=alpha)

            self.print_epoch(start, e, reconst_loss, supervised_loss, total_loss, accuracy, accuracy_topk)

            if e % self.eval_every == 0:
                vali_loss, vali_accuracy, vali_accuracy_topk = self.dev_evaluatorevaluate_iter(encoder=encoder,
                                                                                               decoder=decoder,
                                                                                               classifier=classifier,
                                                                                               input=self.dev_iter,
                                                                                               reconstruction_criterion=reconstruction_criterion,
                                                                                               supervised_criterion=supervised_criterion,
                                                                                               save_path=self.save_path,
                                                                                               topk=self.topk)
                if best_vali_acc < vali_accuracy:
                    best_vali_loss = vali_loss
                    best_vali_acc = vali_accuracy
                    best_vali_acc_topk = vali_accuracy_topk
                    save_best_model(encoder, self.save_path, filename="saved_best_encoder.pt")
                    save_best_model(decoder, self.save_path, filename="saved_best_decoder.pt")
                    save_best_model(classifier, self.save_path, filename="saved_best_classifier.pt")

                self.print_validation(vali_loss, best_vali_loss, vali_accuracy, best_vali_acc, vali_accuracy_topk,
                                      best_vali_acc_topk)

            if e % self.save_every == 0:
                filename = "saved_model_step{}.pt".format(e)
                out_path = os.path.abspath(os.path.join(self.save_path, filename))
                if old_path is not None:
                    os.remove(old_path)
                torch.save({
                    "epoch": e,
                    "best_vali_acc": best_vali_acc,
                    "best_vali_loss": best_vali_loss,
                    "best_vali_acc_topk": best_vali_acc_topk,
                    "encoder_state_dict": encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'classifier_optimizer_state_dict': classifier_optimizer.state_dict()
                }, out_path)
                old_path = out_path

        test_loss, test_accuracy, test_accuracy_topk = self.test_evaluatorevaluate_iter(encoder=encoder,
                                                                                        decoder=decoder,
                                                                                        classifier=classifier,
                                                                                        input=self.dev_iter,
                                                                                        reconstruction_criterion=reconstruction_criterion,
                                                                                        supervised_criterion=supervised_criterion,
                                                                                        save_path=self.save_path,
                                                                                        topk=self.topk)
        self.print_test(test_loss, test_accuracy, test_accuracy_topk)

    def train(self, encoder, decoder, classifier, encoder_optimizer, decoder_optimizer, classifier_optimizer,
              reconst_criterion, supervised_criterion, alpha=1):
        epoch_reconstruction_loss = 0
        epoch_supervised_loss = 0
        epoch_total_acc = 0
        epoch_total_acc_topk = 0
        step = 1

        encoder.train()
        decoder.train()
        classifier.train()

        for batch in self.train_iter:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            batch_x = batch.sentence.to(self.device)
            batch_y = batch.category_labels.to(self.device)

            hidden = encoder(batch_x)
            reconstruction_probs = decoder(hidden)
            supervised_predictions = classifier(hidden.squeeze())

            reconstruction_loss = reconst_criterion(reconstruction_probs, batch_x)
            supervised_loss = supervised_criterion(supervised_predictions, batch_y)

            total_loss = alpha * reconstruction_loss + supervised_loss

            accuracy = calculate_accuracy(supervised_predictions, batch_y)
            accuracy_topk = calculate_topk_accuracy(supervised_predictions, batch_y, topk=self.topk)

            total_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            classifier.step()

            step += 1

            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_supervised_loss += supervised_loss.item()
            epoch_total_acc += accuracy
            epoch_total_acc_topk += accuracy_topk[0].item()

            if step % self.print_every == 0:
                self.print_step(step, reconstruction_loss, supervised_loss, accuracy, accuracy_topk)
            torch.cuda.empty_cache()

        epoch_total_loss = epoch_reconstruction_loss + epoch_supervised_loss
        return epoch_total_loss / len(self.train_iter), epoch_reconstruction_loss / len(
            self.train_iter), epoch_supervised_loss / len(
            self.train_iter), epoch_total_acc / len(self.train_iter), epoch_total_acc_topk / len(self.train_iter)

    def print_step(self, step, reconstruction_loss, supervised_loss, accuracy, accuracy_topk):
        print("Batch {}/{} - "
              "Batch Reconstruction Loss: {:.4f} - "
              "Batch Supervised Loss: {:.4f} - "
              "Batch Accuracy: {:.4f} - "
              "Batch Accuracy Top-{} {:.4f}".format(step,
                                                    len(self.train_iter),
                                                    reconstruction_loss.item(),
                                                    supervised_loss.item(),
                                                    accuracy,
                                                    self.topk[0],
                                                    accuracy_topk[0].item()))

    def print_epoch(self, start, e, reconst_loss, supervised_loss, total_loss, accuracy, accuracy_topk):
        print("{} - "
              "Epoch {}/{} - "
              "Reconstruction Loss: {:.4f} - "
              "Supervised Loss: {:.4f} - "
              "Loss: {:.4f} - "
              "Accuracy: {:.4f} - "
              "Accuracy Top-{}: {:.4f}".format(time_since(start, e / self.epoch),
                                               e,
                                               self.epoch,
                                               reconst_loss,
                                               supervised_loss,
                                               total_loss,
                                               accuracy,
                                               self.topk[0],
                                               accuracy_topk))

    def print_validation(self, vali_loss, best_vali_loss, vali_accuracy, best_vali_acc, vali_accuracy_topk,
                         best_vali_acc_topk):
        print(
            "Validation Loss: {:.4f} (Best: {:.4f}) - "
            "Validation Accuracy: {:.4f} (Best: {:.4f}) - "
            "Validation Accuracy Top-{}: {:.4f} (Best: {:.4f})".format(vali_loss,
                                                                       best_vali_loss,
                                                                       vali_accuracy,
                                                                       best_vali_acc,
                                                                       self.topk[0],
                                                                       vali_accuracy_topk,
                                                                       best_vali_acc_topk))

    def print_test(self, test_loss, test_accuracy, test_accuracy_topk):
        print("Test Loss: {:.4f} - "
              "Test Accuracy: {:.4f} - "
              "Test Accuracy Top-{}: {:.4f}".format(test_loss,
                                                    test_accuracy,
                                                    self.topk[0],
                                                    test_accuracy_topk))

import torch

from utils.utils import calculate_accuracy, calculate_topk_accuracy, load_best_model


class MultipleModelEvaluator(object):
    def __init__(self, device, is_vali):
        self.device = device
        self.is_vali = is_vali

    def evaluate_iter(self, encoder, decoder, classifier, input, reconst_criterion, supervised_criterion, save_path,
                      topk):
        total_loss = 0
        total_acc = 0
        total_acc_topk = 0

        if not self.is_vali:
            print("Test mode!")
            encoder = load_best_model(save_path, filename="saved_best_encoder")
            decoder = load_best_model(save_path, filename="saved_best_decoder")
            classifier = load_best_model(save_path, filename="saved_best_classifier")
        else:
            print("Validation mode!")

        encoder.eval()
        decoder.eval()
        classifier.eval()

        with torch.no_grad():
            for batch in input:
                batch_x = batch.sentence.to(self.device)
                batch_y = batch.category_labels.to(self.device)

                hidden = encoder(batch_x)
                reconstruction_probs = decoder(hidden)
                supervised_predictions = classifier(hidden.squeeze())

                reconstruction_loss = reconst_criterion(reconstruction_probs, batch_x)
                supervised_loss = supervised_criterion(supervised_predictions, batch_y)

                accuracy = calculate_accuracy(supervised_predictions, batch_y)
                accuracy_topk = calculate_topk_accuracy(supervised_predictions, batch_y, topk=topk)

                total_loss = total_loss + reconstruction_loss.item() + supervised_loss.item()
                total_acc += accuracy
                total_acc_topk += accuracy_topk[0].item()

                torch.cuda.empty_cache()

            current_loss = total_loss / len(input)
            current_acc = total_acc / len(input)
            current_acc_topk = total_acc_topk / len(input)

            return current_loss, current_acc, current_acc_topk

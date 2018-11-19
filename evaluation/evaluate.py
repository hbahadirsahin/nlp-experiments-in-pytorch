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

from utils.utils import calculate_accuracy, calculate_topk_accuracy, load_best_model, load_vocabulary


def evaluate_iter(model, input, criterion, device, save_path, topk, is_vali=True):
    total_loss = 0
    total_acc = 0
    total_acc_topk = 0

    if not is_vali:
        print("Test mode!")
        model = load_best_model(save_path)
    else:
        print("Validation mode!")
    model.eval()

    with torch.no_grad():
        for batch in input:
            batch_x = batch.sentence.to(device)
            batch_y = batch.category_labels.to(device)

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)
            accuracy = calculate_accuracy(predictions, batch_y)
            accuracy_topk = calculate_topk_accuracy(predictions, batch_y, topk=topk)

            total_loss += loss.item()
            total_acc += accuracy
            total_acc_topk += accuracy_topk[0].item()

            torch.cuda.empty_cache()

        current_loss = total_loss / len(input)
        current_acc = total_acc / len(input)
        current_acc_topk = total_acc_topk / len(input)

        return current_loss, current_acc, current_acc_topk


def evaluate_interactive(model_path, sentence_vocab_path, category_vocab_path, preprocessor, topk, device="cpu"):
    sentence_vocab = load_vocabulary(sentence_vocab_path)
    category_vocab = load_vocabulary(category_vocab_path)

    model = load_best_model(model_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        while True:
            try:
                sentence = input("Enter a test sentence (Type q or quit to exit!):")
            except ValueError:
                print("Invalid input. Try again! (Type q or quit to exit!)")
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

                tensored_test_sentence = torch.LongTensor(indexed_test_sentence).to(device)

                tensored_test_sentence = tensored_test_sentence.unsqueeze(1)

                logit = model(tensored_test_sentence)
                probs = F.softmax(logit, dim=1)

                predicted_category_probs, predicted_category_ids = probs.topk(topk, 1, True, True)

                predicted_category_ids = predicted_category_ids.t()

                predicted_labels = []
                for idx in predicted_category_ids:
                    predicted_labels.append(category_vocab.itos[idx])

                if topk == 1:
                    print("Predicted category is {} with probability {}".format(predicted_labels[0],
                                                                                predicted_category_probs[0][0].item()))
                else:
                    print("Top-{} predicted labels are as follows in order:".format(topk))
                    for idx, label in enumerate(predicted_labels):
                        print("> {} - Predicted category is {} with probability {:.4f}".format(idx + 1,
                                                                                               label,
                                                                                               predicted_category_probs[
                                                                                                   0][idx].item()))
            else:
                print("Interactive evaluation ends!")
                break

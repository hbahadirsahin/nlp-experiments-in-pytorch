import torch

from utils.utils import calculate_accuracy, load_best_model, load_vocabulary


def evaluate_iter(model, input, criterion, device, save_path, is_vali=True):
    total_loss = 0
    total_acc = 0

    if not is_vali:
        print("Test mode!")
        model = load_best_model(save_path)

    print("Validation mode!")
    model.eval()

    with torch.no_grad():
        for batch in input:
            batch_x = batch.sentence.to(device)
            batch_y = batch.category_labels.to(device)

            predictions = model(batch_x).squeeze(1)

            loss = criterion(predictions, batch_y)
            accuracy = calculate_accuracy(predictions, batch_y)

            total_loss += loss.item()
            total_acc += accuracy

            torch.cuda.empty_cache()

        current_loss = total_loss / len(input)
        current_acc = total_acc / len(input)

        return current_loss, current_acc


def evaluate_interactive(model_path, sentence_vocab_path, category_vocab_path, preprocessor, device="cpu"):
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
                preprocessed_sentence = preprocessor.preprocess(sentence.split())
                indexed_test_sentence = [sentence_vocab.stoi[token] for token in preprocessed_sentence]

                tensored_test_sentence = torch.LongTensor(indexed_test_sentence).to(device)

                tensored_test_sentence = tensored_test_sentence.unsqueeze(1)

                logit = model(tensored_test_sentence)

                predicted_category_id = torch.max(logit, 1)[1]

                predicted_category_label = category_vocab.itos[predicted_category_id]

                print("Predicted category is {}".format(predicted_category_label))
            else:
                print("Interactive evaluation ends!")
                break

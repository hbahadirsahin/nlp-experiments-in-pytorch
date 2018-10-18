import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils as utils
import time
import os
from utils.utils import time_since, calculate_accuracy, save_best_model
from evaluation.evaluate import evaluate_iter


def init_optimizer(optimizer_type, model, learning_rate, weight_decay, momentum):
    print("Optimizer type is {} !".format(optimizer_type))

    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise KeyError("Invalid optimizer type! Choose Adam or SGD!")


def train(model, train_iter, optimizer, scheduler, criterion, norm_ratio, device, print_every=500):
    epoch_total_loss = 0
    epoch_total_acc = 0
    step = 1
    model.train()
    for batch in train_iter:
        optimizer.zero_grad()

        batch_x = batch.sentence.to(device)
        batch_y = batch.category_labels.to(device)

        predictions = model(batch_x).squeeze(1)

        loss = criterion(predictions, batch_y)
        accuracy = calculate_accuracy(predictions, batch_y)

        loss.backward()

        utils.clip_grad_norm_(model.parameters(), max_norm=norm_ratio)

        optimizer.step()
        if scheduler is not None and step % 500 == 0:
            scheduler.step(step)

        step += 1

        epoch_total_loss += loss.item()
        epoch_total_acc += accuracy

        if step % print_every == 0:
            print("Batch {}/{} - Batch Loss: {:.4f} - Batch Accuracy: {:.4f}".format(step,
                                                                                     len(train_iter),
                                                                                     loss,
                                                                                     accuracy))
        torch.cuda.empty_cache()

    return epoch_total_loss / len(train_iter), epoch_total_acc / len(train_iter)


def train_iters(model, train_iter, dev_iter, test_iter, device, training_properties):
    optimizer_type = training_properties["optimizer"]
    learning_rate = training_properties["learning_rate"]
    weight_decay = training_properties["weight_decay"]
    momentum = training_properties["momentum"]
    norm_ratio = training_properties["norm_ratio"]
    epoch = training_properties["epoch"]
    print_every = training_properties["print_every_batch_step"]
    save_every = training_properties["save_every_epoch"]
    eval_every = training_properties["eval_every"]
    save_path = training_properties["save_path"]

    optimizer = init_optimizer(optimizer_type, model, learning_rate, weight_decay, momentum)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    criterion = nn.CrossEntropyLoss().to(device)

    start = time.time()
    old_path = None
    best_vali_acc = -1

    print("Training...")
    for e in range(1, epoch + 1):
        loss, accuracy = train(model=model,
                               train_iter=train_iter,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               criterion=criterion,
                               norm_ratio=norm_ratio,
                               device=device,
                               print_every=print_every)

        print("{} - Epoch {}/{} - Loss: {:.4f} - Accuracy: {:.4f}".format(time_since(start, e / (len(train_iter))),
                                                                          e,
                                                                          epoch,
                                                                          loss,
                                                                          accuracy))

        if e % save_every == 0:
            filename = "saved_model_step{}.pt".format(e)
            out_path = os.path.abspath(os.path.join(save_path, filename))
            if old_path is not None:
                os.remove(old_path)
            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, out_path)
            old_path = out_path

        if e % eval_every == 0:
            vali_loss, vali_accuracy = evaluate_iter(model=model,
                                                     input=dev_iter,
                                                     criterion=criterion,
                                                     device=device,
                                                     save_path=save_path,
                                                     is_vali=True)
            if best_vali_acc < vali_accuracy:
                best_vali_loss = vali_loss
                best_vali_acc = vali_accuracy
                save_best_model(model, save_path)
            print(
                "Validation Loss: {:.4f} (Best: {:.4f}) - Validation Accuracy: {:.4f} (Best: {:.4f})".format(vali_loss,
                                                                                                             best_vali_loss,
                                                                                                             vali_accuracy,
                                                                                                             best_vali_acc))

    test_loss, test_accuracy = evaluate_iter(model=model,
                                             input=test_iter,
                                             criterion=criterion,
                                             device=device,
                                             save_path=save_path,
                                             is_vali=False)
    print("Test Loss: {:.4f} - Test Accuracy: {:.4f}".format(test_loss,
                                                             test_accuracy))

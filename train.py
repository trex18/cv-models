import torch
import time
import tqdm

def train(model, num_epochs, dataloaders, batch_size, optimizer, criterion, device):
    accuracy_history = {phase : [] for phase in ['train', 'val']}
    loss_history = {phase : [] for phase in ['train', 'val']}
    best_epoch = 0
    best_val_loss = 10**10
    best_wts = model.state_dict()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("=" * 80)
        print("Epoch {} / {}".format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            progress_bar = tqdm.tqdm(total=len(dataloaders[phase]), desc=phase, position=0, leave=True)
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0

            for i, (data, targets) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                data = data.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outs = model(data)
                    loss = criterion(outs, targets)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    preds = torch.argmax(outs, dim=1)
                    running_corrects += (preds == targets).sum().detach().item()
                progress_bar.update(1)

            epoch_loss = running_loss / (i * batch_size) #considering that the reduction in criterion is sum
            epoch_accuracy = 100.0 * running_corrects / ((i + 1) * batch_size)
            loss_history[phase].append(epoch_loss)
            accuracy_history[phase].append(epoch_accuracy)


            print("\n%5s loss:\t %.3f" % (phase, epoch_loss))
            print("\n%5s accuracy:\t %.2f%%" % (phase, epoch_accuracy))

        if epoch_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = epoch_loss
            best_wts = model.state_dict()
            print("Best new model!")

        print("Epoch time: %.3f" % (time.time() - start_time))

    model.load_state_dict(best_wts)
    return model, accuracy_history, loss_history
import torch
import torchvision
import time
import copy

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        state = {
            'last_epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, 'model_next' + str(epoch+1) + '.pt')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                torch.cuda.empty_cache()
                inputs = inputs.view(inputs.shape[0]*10, 3, 224, 224) #batch
                labels = labels.view(labels.shape[0]*10,1) #batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = normalize(model(inputs))
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        scheduler.step()
        if epoch>18 and (epoch+1)%5==0:
            #save
            state = {
                'last_epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, 'model_next' + str(epoch+1) + '.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def get_target():
    return torch.tensor([1,1,0,0,0,0,0,0,0,0], dtype=torch.float32)

#onehot
def twohot_decode(X):
    return torch.argmax(torch.narrow(X, 0, 2, 8))

def normalize(X):
    return torch.sigmoid(X)

def freeze_BatchNorm2d(model):
    for name, child in (model.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
        else:
            if isinstance(child, torch.nn.Sequential):
                for seqname, seqchild in (child.named_children()):
                    if isinstance(seqchild, torchvision.models.resnet.BasicBlock):
                        for mininame, minichild in (seqchild.named_children()):
                            if isinstance(minichild, torch.nn.BatchNorm2d):
                                for param in minichild.parameters():
                                    param.requires_grad = False
                            else:
                                if isinstance(minichild, torch.nn.Sequential):
                                    for seqmininame, seqminichild in (minichild.named_children()):
                                        if isinstance(seqminichild, torch.nn.BatchNorm2d):
                                            for param in seqminichild.parameters():
                                                param.requires_grad = False
                                        else:
                                            for param in seqminichild.parameters():
                                                param.requires_grad = True
                                else:
                                    for param in minichild.parameters():
                                        param.requires_grad = True
                    else:
                        for param in seqchild.parameters():
                            param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = True
    return


#load
'''
state = torch.load('model_nextXX.pt')
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
scheduler.load_state_dict(state['scheduler'])
next_epoch = state['last_epoch']+1
'''







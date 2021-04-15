import torch
import torchvision
import time
import copy

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                scheduler.step()
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0
            corrects = 0
            
            # Iterate over data.
            for inputs, labels, answers in dataloaders[phase]:
                batch = inputs.shape[0]
                inputs = inputs.view(batch*10, 3, 224, 224)                  
                labels = labels.view(batch*10,1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                answers = answers.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = normalize(model(inputs))
                    loss = criterion(outputs, labels)
                    
                    outputs = outputs.view(batch,10)
                    preds = torch.round(outputs)
                    labels = labels.view(batch,10)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds.data==labels.data)
                corrects += torch.sum(answers.data==twohot_decode(outputs))
                
            epoch_loss = running_loss / (len(dataloaders[phase].dataset)*10)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset)*10) * 100
            real_acc = corrects.double() / len(dataloaders[phase].dataset) * 100

            print('{} Loss: {:.4f} Acc: {:.4f}%'.format(phase, epoch_loss, epoch_acc))
            print('Real Acc: {:.4f}%'.format(real_acc))
            
            # deep copy the model
            if phase == 'val' and real_acc > best_acc:
                best_acc = real_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(real_acc.item())

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
    return torch.argmax(torch.narrow(X, 1, 2, 8), dim=1)

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
    return

def test_model(device, model, dataloaders):
    since = time.time()

    #load
    state = torch.load('LR_single_best.pt', device)
    model.load_state_dict(state)
    model.to(device)

    print('Testing...')
    print('-' * 10)

    # Each epoch has a training and validation phase
    phase = 'test'
    model.eval()   # Set model to evaluate mode
    corrects = 0
    
    # Iterate over data.
    for inputs, labels, answers in dataloaders[phase]:
        batch = inputs.shape[0]
        inputs = inputs.view(batch*10, 3, 224, 224)
        inputs = inputs.to(device)
        answers = answers.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            outputs = normalize(model(inputs))

            preds = twohot_decode(outputs.view(batch,10))
            
            
        # statistics
        corrects += torch.sum(preds.data==answers.data)

    real_acc = corrects.double() / len(dataloaders[phase].dataset) * 100
    print('Acc: {:.4f}%'.format(real_acc))

    print()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return

#load
'''
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss') #train_losses.append(loss)
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
'''


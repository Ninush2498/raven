import torch
import torchvision
import time
import copy

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    val_acc_history = []
    model.to(device)

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
                optimizer.step() #last incomplete batch
                optimizer.zero_grad() #acc
                scheduler.step()
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects1 = 0
            running_corrects0 = 0
            corrects = 0
            i = 0 #acc
            
            # Iterate over data.
            for inputs, labels, answers in dataloaders[phase]:
                torch.cuda.empty_cache()
                inputs = inputs.view(inputs.shape[0]*10, 3, 224, 224) #batch
                labels = labels.view(labels.shape[0]*10,1) #batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                answers = answers.to(device)

                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = normalize(model(inputs))
                    loss = criterion(outputs, labels)

                    preds = torch.round(outputs).view(int(list(outputs.shape)[0]/10),10)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if (i+1) % 8 == 0: #accumulate
                            optimizer.step()
                            optimizer.zero_grad()
                        i += 1

                        #optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects1 += torch.sum(preds[:,:2].data == labels.view(int(list(labels.shape)[0]/10),10)[:,:2].data)
                running_corrects0 += torch.sum(preds[:,2:].data == labels.view(int(list(labels.shape)[0]/10),10)[:,2:].data)
                corrects += torch.sum(answers.data == twohot_decode(outputs.view(int(list(outputs.shape)[0]/10),10)))

            epoch_loss = running_loss / (len(dataloaders[phase].dataset)*10)
            epoch_acc1 = running_corrects1.double() / (len(dataloaders[phase].dataset)*2) * 100
            epoch_acc0 = running_corrects0.double() / (len(dataloaders[phase].dataset)*8) * 100
            real_acc = corrects.double() / len(dataloaders[phase].dataset) * 100

            print('{} Loss: {:.4f} Acc1: {:.4f}% Acc0: {:.4f}%'.format(phase, epoch_loss, epoch_acc1, epoch_acc0))
            print('Real Acc: {:.4f}%'.format(real_acc))
            
            # deep copy the model
            if phase == 'val' and real_acc > best_acc:
                best_acc = real_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(real_acc)
            
        if (epoch+1)%5==0:
            #save
            torch.save(best_model_wts, 'center_single_after' + str(epoch) + '.pt')

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

def test_model(device, model, dataloaders):
    since = time.time()

    #load
    state = torch.load('centre_single_after4.pt')
    model.load_state_dict(state['state_dict'])
    model.to(device)

    print('Testing...')
    print('-' * 10)

    # Each epoch has a training and validation phase
    phase = 'test'
    model.eval()   # Set model to evaluate mode
    corrects = 0
    
    # Iterate over data.
    for inputs, labels, answers in dataloaders[phase]:
        torch.cuda.empty_cache()
        inputs = inputs.view(inputs.shape[0]*10, 3, 224, 224) #batch
        inputs = inputs.to(device)
        answers = answers.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            outputs = normalize(model(inputs))

            preds = twohot_decode(outputs.view(int(list(outputs.shape)[0]/10),10))

        # statistics
        corrects += torch.sum(answers.data == preds.data)

    real_acc = corrects.double() / len(dataloaders[phase].dataset) * 100
    print('Acc: {:.4f}%'.format(real_acc))

    print()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return

#load
'''
state = torch.load('model_nextXX.pt')
model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])
scheduler.load_state_dict(state['scheduler'])
next_epoch = state['last_epoch']+1
'''










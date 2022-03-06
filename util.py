import torch
import torchvision
import time
import copy
import network as n

def train_model(device, model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    val_acc_history = []
    
    #make a copy of the model to keep track of the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #training and validation phase in each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  #set model to training mode
            else:
                scheduler.step()
                model.eval()   #set model to evaluation mode

            running_loss = 0.0
            corrects = 0
            
            #iterate over data
            for inputs, labels in dataloaders[phase]:
                #GPU computations
                inputs = inputs.to(device)
                labels = labels.to(device)

                #zero the parameter gradients
                optimizer.zero_grad()

                #forward pass
                #track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    #get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)

                    #backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                #statistics
                running_loss += loss.item()*inputs.size(0)
                corrects += torch.sum(preds==labels)
            
            epoch_loss = running_loss / (len(dataloaders[phase].dataset))
            real_acc = corrects.double() / (len(dataloaders[phase].dataset)) * 100

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('Real Acc: {:.4f}%'.format(real_acc))
            
            #deep copy the model if better
            if phase == 'val' and real_acc > best_acc:
                best_acc = real_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(real_acc.item())

        print()

    time_elapsed = time.time() - since #track the training time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights and return the best model
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

#freeze batch normalization layers of ResNet-18
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

def test_model(device, model, dataloaders, visualize=False, batch_size=1):
    since = time.time()

    #load model (not necessary when testing directly after training)
    state = torch.load('all_best.pt', device)
    model.load_state_dict(state)
    
    if visualize: #add data collecting layer
        model.fc.add_module('output', model.fc[2])
        model.fc[2] = n.Activations(len(dataloaders['test'].dataset), batch_size)
    model.to(device)

    print('Testing...')
    print('-' * 10)

    #only testing phase
    phase = 'test'
    model.eval()   #set model to evaluation mode
    corrects = 0
    
    #iterate over data
    for inputs, labels, set_type in dataloaders[phase]:
        torch.cuda.empty_cache()
        if visualize: #collect types
            model.fc[2].add_labels(set_type)
        inputs = inputs.to(device)
        labels = labels.to(device)

        #forward pass
        #does not track history
        with torch.set_grad_enabled(phase == 'train'):
            #get model outputs and calculate loss
            outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            
            if visualize: #collect answers
                model.fc[2].add_outputs(preds)
        
        
        #statistics
        corrects += torch.sum(preds==torch.argmax(labels, dim=1))

    real_acc = corrects.double() / len(dataloaders[phase].dataset) * 100
    print('Acc: {:.4f}%'.format(real_acc))

    print()

    if visualize:
        model.fc[2].visualize()

    time_elapsed = time.time() - since #track testing time
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return

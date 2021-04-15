import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from util import *

def get_nn(device=torch.device('cuda'), pretrained=False):
    model = models.resnet18(pretrained)
        
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512,1))
    
    freeze_BatchNorm2d(model)
    model.to(device)

    loss = nn.BCELoss('sum')

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    change_rate = lambda epoch : 0.5 if epoch%10==0 else 1
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, change_rate)

    return model, loss, optimizer, scheduler


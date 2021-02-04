#zakomentovane poznamky by mali byt implementovane
#nezakomentovane este treba pridat

#Poznamky:
#fully connected layer randomly initialized
#extra dropout layer before fully connected - default value 0.5
#mini-batch = 32
#Adam optimizer
#rate 0.0002, reducing to half every 10 iterations
#images 224 x 224, normalized to range <0,1>
#6 training sets, 2 validation, 2 testing
#freeze all batch normalization layers during training
#binary cross entropy
#sigmoid for normalization

#Kod:
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from util import *

def get_nn(device, pretrained=False):
    model = models.resnet18()
    model_trained = models.resnet18(pretrained=True) #tento pouzili

    if pretrained:
        model = model_trained
        
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 1))

    freeze_BatchNorm2d(model)

    batch_size = 32
    feature_extract = False
    loss = nn.BCELoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    lambda_function = lambda epoch : 1 if epoch%10==0 else 0.5
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda_function)

    return model, loss, optimizer, scheduler






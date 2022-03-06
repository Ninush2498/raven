import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import umap
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from util import *

#get model
def get_nn(device=torch.device('cuda'), pretrained=False):
    model = models.resnet18(pretrained)

    #add dropout before the output layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512,8)) #8 output neurons

    #reshape the 
    model.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False) #sup
    
    freeze_BatchNorm2d(model)
    model.to(device) #for GPU computations

    loss = nn.CrossEntropyLoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    change_rate = lambda epoch : 0.5 if epoch%10==0 else 1
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, change_rate)

    return model, loss, optimizer, scheduler

#class that collects the activations on the penultimate layer for visualization
class Activations(nn.Module):

    def __init__(self, size, batch_size):
        self.activations = torch.zeros(size, 100) #100 hidden neurons
        self.labels = []
        self.outputs = torch.zeros(size)
        self.counter = 0
        self.batch_size = batch_size
        self.types = []
        super().__init__()
        
    #collect the outputs and do not modify them
    def forward(self, x):
        self.activations[self.counter:self.counter+self.batch_size] = x
        self.counter += self.batch_size
        return x

    #collect the types (subsets) of the outputs for plotting
    def add_labels(self, labels):
        for l in labels:
            if l not in self.labels:
                self.types.append(l)
            self.labels.append(l)
        return

    #collect the answers (which candidate was selected) for plotting
    def add_outputs(self, outs):
        self.outputs[self.counter-self.batch_size:self.counter] = outs
        return

    #plotting
    def visualize(self):
        reducer = umap.UMAP()
        reducer.fit(self.activations.data)
        embedding = reducer.transform(self.activations.data)
        assert(np.all(embedding == reducer.embedding_))
        
        C = []
        cC = []
        x2 = []
        cx2 = []
        x3 = []
        cx3 = []
        LR = []
        cLR = []
        UD = []
        cUD = []
        inC = []
        cinC = []
        in2x2 = []
        cin2x2 = []
        colors = lambda x: 'red' if x==0 else 'blue' if x==1 else 'black' if x==2 else 'green' if x==3 else 'purple' if x==4 else 'chocolate' if x==5 else 'lawngreen' if x==6 else 'aqua'
        cols = list(map(colors, self.outputs))
        for i in range(len(embedding)):
            e = embedding[i]
            c = cols[i]
            if self.labels[i]=='C':
                C.append(e)
                cC.append(c)
            elif self.labels[i]=='2x2':
                x2.append(e)
                cx2.append(c)
            elif self.labels[i]=='3x3':
                x3.append(e)
                cx3.append(c)
            elif self.labels[i]=='LR':
                LR.append(e)
                cLR.append(c)
            elif self.labels[i]=='UD':
                UD.append(e)
                cUD.append(c)
            elif self.labels[i]=='inC':
                inC.append(e)
                cinC.append(c)
            elif self.labels[i]=='in2x2':
                in2x2.append(e)
                cin2x2.append(c)
        C = torch.tensor(C)
        x2 = torch.tensor(x2)
        x3 = torch.tensor(x3)
        LR = torch.tensor(LR)
        UD = torch.tensor(UD)
        inC = torch.tensor(inC)
        in2x2 = torch.tensor(in2x2)
            

        fig, axs = plt.subplots(3, 3)

        axs[0,0].scatter(C[:,0],C[:,1], s=3, c=cC)
        axs[0,0].set_title('C')
        axs[0,1].scatter(x2[:,0], x2[:,1], s=3, c=cx2)
        axs[0,1].set_title('2x2')
        axs[0,2].scatter(x3[:,0], x3[:,1], s=3, c=cx3)
        axs[0,2].set_title('3x3')
        axs[1,0].scatter(LR[:,0], LR[:,1], s=3, c=cLR)
        axs[1,0].set_title('LR')
        axs[1,1].scatter(UD[:,0], UD[:,1], s=3, c=cUD)
        axs[1,1].set_title('UD')
        axs[1,2].scatter(inC[:,0], inC[:,1], s=3, c=cinC)
        axs[1,2].set_title('inC')
        axs[2,0].scatter(in2x2[:,0], in2x2[:,1], s=3, c=cin2x2)
        axs[2,0].set_title('in2x2')
        axs.flat[-1].set_visible(False)
        axs.flat[-2].set_visible(False)
        for a in axs.flat:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        plt.show()
        return

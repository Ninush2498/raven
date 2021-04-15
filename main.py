import numpy as np
import torchvision
import torch
from network import *
from util import *

class Dataset(torch.utils.data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, phase):
        'Initialization'
        self.list_IDs = list_IDs
        self.phase = phase #'train', 'val', 'test'
        self.p = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((224,224),interpolation=3)])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        if ID[0]=="UD":
            data = np.load('RAVEN-10000/up_center_single_down_center_single/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="LR":
            data = np.load('RAVEN-10000/left_center_single_right_center_single/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="2x2":
            data = np.load('RAVEN-10000/distribute_four/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="C":
            data = np.load('RAVEN-10000/center_single/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="3x3":
            data = np.load('RAVEN-10000/distribute_nine/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="in2x2":
            data = np.load('RAVEN-10000/in_distribute_four_out_center_single/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        elif ID[0]=="inC":
            data = np.load('RAVEN-10000/in_center_single_out_center_single/RAVEN_' + ID[1] + '_' + self.phase + '.npz')
        X = torch.ones(size=(16, 224, 224))
        for i in range(16):
            X[i] = self.p(data['image'][i])
        X = X.unsqueeze(dim=0)
        
        ans = data['target']
        y = get_target()

        #images of the choices
        choices = X[:, 8:].unsqueeze(dim=2) #[b, 8, 1, h, w]

        #images of the rows
        row1 = X[:, 0:3].unsqueeze(1) #[b, 1, 3, h, w]
        row2 = X[:, 3:6].unsqueeze(1) #[b, 1, 3, h, w]

        row3_p = X[:, 6:8].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1) #[b, 8, 2, h, w]
        row3 = torch.cat((row3_p, choices), dim=2) #[b, 8, 3, h, w]

        rows = torch.cat((row1, row2, row3), dim=1).squeeze(dim=0) #[b, 10, 3, h, w]
        

        return rows, y, ans

dataset = []
dataset.append('C')
dataset.append('2x2')
dataset.append('3x3')
dataset.append('LR')
dataset.append('UD')
dataset.append('inC')
dataset.append('in2x2')
partition = {'train': [(dataset[i],str(x*10+y)) for i in range(7) for x in range(1000) for y in range(6)], # + [(dataset2,str(x*10+y)) for x in range(1000) for y in range(6)] + [(dataset3,str(x*10+y)) for x in range(1000) for y in range(6)],
             'val': [(dataset[i],str(x*10+y)) for i in range(7) for x in range(1000) for y in range(6,8)], # + [(dataset2,str(x*10+y)) for x in range(1000) for y in range(6,8)] + [(dataset3,str(x*10+y)) for x in range(1000) for y in range(6,8)],
             'test': [(dataset[i],str(x*10+y)) for i in [1,6] for x in range(1000) for y in range(8,10)]} # + [(dataset2,str(x*10+y)) for x in range(1000) for y in range(8,10)]} # + [(dataset3,str(x*10+y)) for x in range(1000) for y in range(8,10)]}


train_set = Dataset(partition['train'],'train')
val_set = Dataset(partition['val'],'val')
test_set = Dataset(partition['test'],'test')

data_loader = {'train': torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True),
               'val': torch.utils.data.DataLoader(val_set,batch_size=32,shuffle=False),
               'test': torch.utils.data.DataLoader(test_set,batch_size=16,shuffle=False)}

#train
epochs = 30
print('Training:')
device = torch.device('cuda')
model, loss, optimizer, scheduler = get_nn(device,True)

best_model, val_acc_history = train_model(device,model,data_loader,loss,optimizer,scheduler,epochs)
torch.save(best_model.state_dict(), 'all_unsup.pt')

print(val_acc_history)

#test_model(device, model, data_loader)


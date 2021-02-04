import numpy as np
import torchvision
import torch
from network import *

class Dataset(torch.utils.data.Dataset):
    
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, phase):
        'Initialization'
        self.list_IDs = list_IDs
        self.phase = phase #'train', 'val', 'test'
        self.p = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((224,224))])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        data = np.load('RAVEN-10000/center_single/RAVEN_' + ID + '_' + self.phase + '.npz')
        X = torch.zeros(size=(16, 224, 224))
        for i in range(16):
            X[i] = self.p(data['image'][i])
        X = X.unsqueeze(dim=0)

        #images of the choices
        choices = X[:, 8:].unsqueeze(dim=2) #[b, 8, 1, h, w]

        #images of the rows
        row1 = X[:, 0:3].unsqueeze(1) #[b, 1, 3, h, w]
        row2 = X[:, 3:6].unsqueeze(1) #[b, 1, 3, h, w]

        row3_p = X[:, 6:8].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1) #[b, 8, 2, h, w]
        row3 = torch.cat((row3_p, choices), dim=2) #[b, 8, 3, h, w]

        rows = torch.cat((row1, row2, row3), dim=1).squeeze(dim=0) #[b, 10, 3, h, w]

        #y = data['target']
        y = get_target()

        return rows, y



partition = {'train': [str(x*10+y) for x in range(100) for y in range(6)],
             'val': [str(x*10+y) for x in range(1000) for y in range(6,8)],
             'test': [str(x*10+y) for x in range(1000) for y in range(8,10)]}

'''
#input shape b*16*224*224, 16 images with a size of 224*224
b = x.shape[0]   #b is batch size

#images of the choices
choices = x[:, 8:].unsqueeze(dim=2) #[b, 8, 1, h, w]
print()
#images of the rows
row1 = x[:, 0:3].unsqueeze(1) #[b, 1, 3, h, w]
row2 = x[:, 3:6].unsqueeze(1) #[b, 1, 3, h, w]

row3_p = x[:, 6:8].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1) #[b, 8, 2, h, w]
row3 = torch.cat((row3_p, choices), dim=2) #[b, 8, 3, h, w]

rows = torch.cat((row1, row2, row3), dim=1) #[b, 10, 3, h, w]

#x = rows.view(b*10, 3, 224, 224) #reshape the input for training, use same the weights for 10 rows
#print(x)
#x = feature(x)
#x = fc(x) #fully connected layer, output dimension is 1
#x = x.view(b, 10) #output size is b*10
'''
train_set = Dataset(partition['train'],'train')
val_set = Dataset(partition['val'],'val')
data_loader = {'train': torch.utils.data.DataLoader(train_set,batch_size=4,shuffle=True), 'val': torch.utils.data.DataLoader(val_set,batch_size=32,shuffle=True)}

#train
device = torch.device('cuda')
model, loss, optimizer, scheduler = get_nn(device)
best_model, val_acc_history = train_model(device,model,data_loader,loss,optimizer,scheduler,20,is_inception=False)







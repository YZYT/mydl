import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# For data preprocess
import numpy as np
import csv
import os
import argparse

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def init_Seed(myseed=42069):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if get_device() == 'cuda':
        torch.cuda.manual_seed_all(myseed)
        print("GPU ready!") 
    else:
        print("No GPU!")


class MLDataset(Dataset):
    """ Dataset for loading and preprocessing the MoviesLen dataset. """
    def __init__(self, path, mode='train', target_only=False):
        super().__init__()
        self.mode = mode

        # Read data into numpy arrays
        data = np.loadtxt(path, dtype='long')

        # Convert data into PyTorch tensors
        self.data = torch.LongTensor(data[:, :2])
        self.target = torch.FloatTensor(data[:, 2])

        print(f'Max user: {max(self.data[:, 0])}')
        print(f'Max item: {max(self.data[:, 1])}')

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of MoviesLen Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))
    

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]
    

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


class MF(nn.Module):
    def __init__(self, n_users=6050, m_items=4050, n_factors=20, alpha=0.05):

        super(MF, self).__init__()
        self.alpha = alpha
        self.U = torch.nn.Parameter(torch.rand(n_users, n_factors, requires_grad=True))
        self.V = torch.nn.Parameter(torch.rand(m_items, n_factors, requires_grad=True))

        self.U.data.uniform_(-0.005, 0.005)
        self.V.data.uniform_(-0.005, 0.005)
        
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, user, item):
        # return torch.einsum('ij, ij -> i', [self.U[user], self.P[item] @ self.Q])
        return torch.einsum('ij, ij -> i', [self.U[user], self.V[item]])

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        return self.criterion(pred, target)

    def regularizer(self, user, item):
        return self.alpha * (self.U[user].norm() ** 2 + self.V[item].norm() ** 2)


class SLMF(nn.Module):
    def __init__(self, n_users=1050, m_items=2001, n_factors=20):
        super(MF, self).__init__()

        self.U = torch.nn.Parameter(torch.rand(n_users, n_factors, requires_grad=True))

        self.P = torch.rand(m_items, 1000 * n_factors, requires_grad=False, device=device)
        
        self.net = nn.Sequential(nn.Linear(1000 * n_factors, n_factors), nn.ReLU(), nn.Linear(n_factors, n_factors))


        self.init_net()

        self.U.data.uniform_(-0.005, 0.005)
        
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, user, item):
        # return torch.einsum('ij, ij -> i', [self.U[user], self.P[item] @ self.Q])
        return torch.einsum('ij, ij -> i', [self.U[user], self.net(self.P[item])])

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        return self.criterion(pred, target)
    
    def init_net(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        self.net.apply(init_weights)


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = MLDataset(path, mode=mode)  # Construct dataset
    dataloader = DataLoader(dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


def train(tr_set, dv_set, model, optimizer, scheduler=None, device='cpu', n_epochs=100, early_stop=5):
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []} 
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        model.train()
        for X, y in tr_set:
            optimizer.zero_grad()    
            X, y = X.to(device), y.to(device)
            y_hat = model(X[:, 0], X[:, 1])

            mse_loss = model.cal_loss(y_hat, y) + model.regularizer(X[:, 0], X[:, 1]) / len(y)
            
            mse_loss.backward()
            
            optimizer.step()

            loss_record['train'].append(mse_loss.detach().cpu().item())

            # print("train_loss: {:.4f}".format(np.sqrt(mse_loss.detach().cpu().item())))

        if scheduler:
            scheduler.step()
        
        epoch += 1

        dev_mse = dev(dv_set, model, device)

        print("epoch = {:4d} dev_loss: {:.4f}".format(epoch, np.sqrt(dev_mse)))

        if dev_mse < min_mse:
            min_mse = dev_mse
            early_stop_cnt = 0
            print("Saving model (epoch = {:4d}  loss = {:.4f} )".format(epoch, np.sqrt(dev_mse)))
            # torch.save(model.state_dict(), config['save_path'])
        else:
            early_stop_cnt += 1
        
        
        loss_record['dev'].append(dev_mse)

        if early_stop_cnt > early_stop:
            break

    print("Finish training after {} epochs".format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for X, y in dv_set:                         # iterate through the dataloader
        X, y = X.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(X[:, 0], X[:, 1])                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(y)    # accumulate loss
        # total_loss += mse_loss.detach().cpu().item()     # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds


def readConfig():
    config = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-n_epochs', type=int)
    parser.add_argument('-batch_size', type=int)
    parser.add_argument('-alpha', type=float)
    parser.add_argument('-optimizer', type=str)
    parser.add_argument('-lr', type=float)
    parser.add_argument('-weight_decay', type=float)
    parser.add_argument('-step_size', type=int)
    parser.add_argument('-gamma', type=float)
    parser.add_argument('-early_stop', type=int)
    parser.add_argument('-save_path', type=str)
    parser.add_argument('-D', type=int)

    args = parser.parse_args()

    config['dataset'] = args.dataset
    config['n_epochs'] = args.n_epochs
    config['batch_size'] = args.batch_size
    config['alpha'] = args.alpha
    config['optimizer'] = args.optimizer
    optim_hparas = {}
    optim_hparas['lr'] = args.lr
    optim_hparas['weight_decay'] = args.weight_decay
    config['optim_hparas'] = optim_hparas
    config['step_size'] = args.step_size
    config['gamma'] = args.gamma
    config['early_stop'] = args.early_stop
    config['save_path'] = args.save_path
    config['D'] = args.D

    print(config)
    return  config

    # 'n_epochs': 500,              # maximum number of epochs
    # 'batch_size': 500,               # mini-batch size for dataloader
    # 'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    # 'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
    #     'lr': 0.001,                # learning rate
    #     'weight_decay': 0.0001
    #     # 'momentum': 0.9              # momentum for SGD
    # },
    # 'early_stop': 5,               # early stopping epochs (the number epochs since your model's last improvement)
    # 'save_path': 'models/model.pth',  # your model will be saved here
    # 'D': 50



def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()







def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


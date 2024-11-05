from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import os
import csv
import copy
import pandas as pd
from tqdm import tqdm
from scipy import stats
from inceptionresnetv2 import inceptionresnetv2

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# from clearml import Task, Logger


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    return pil_loader(path)


class Koniq_10k(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.imgname = []
        self.mos = []
        self.csv_file = os.path.join(self.root, 'koniq10k_scores_and_distributions.csv')
        with open(self.csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgname.append(row['image_name'])
                mos = float(row['MOS'])
                mos = np.array(mos)
                mos = mos.astype(np.float32)
                self.mos.append(mos)

        sample = []

        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, '1024x768', self.imgname[item]), self.mos[item]))
        self.samples = sample
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class ModelManager:
    def __init__(self, model, batch_size=16, lr=1e-4, train_index=None, val_index=None, test_index=None, device='cuda'):
        assert train_index and val_index and test_index
        self.device = device
        self.model = model.to(self.device)
        self.lr = lr
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop((384,512), ),
                #transforms.Resize((384,512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ]),
            'val': transforms.Compose([
                # transforms.RandomResizedCrop(((384,512)), ),
                transforms.Resize((384,512)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self.batch_size = batch_size
        dataset_path = os.path.join('dataset', 'KonIQ-10k')
        train_data = Koniq_10k(
                root=dataset_path, loader=default_loader, index=train_index,
                transform=self.data_transforms['train'])
        val_data = Koniq_10k(
                root=dataset_path, loader=default_loader, index=val_index,
                transform=self.data_transforms['val'])
        test_data = Koniq_10k(
                root=dataset_path, loader=default_loader, index=test_index,
                transform=self.data_transforms['val'])
        
        self._train_loader = data.DataLoader(
            train_data, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True)
        self._val_loader = data.DataLoader(
            val_data, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)

        
    def plcc(self, x, y):
        """Pearson Linear Correlation Coefficient"""
        x, y = np.float32(x), np.float32(y)
        return stats.pearsonr(x,y)[0]

    def srocc(self, xs, ys):
        """Spearman Rank Order Correlation Coefficient"""
        xranks = pd.Series(xs).rank()    
        yranks = pd.Series(ys).rank()    
        return self.plcc(xranks, yranks)
    def rating_metrics(self, y_true, y_pred, show_plot=True):    
        """
        Print out performance measures given ground-truth (`y_true`) and predicted (`y_pred`) scalar arrays.
        """
        y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
        p_plcc = np.round(self.plcc(y_true, y_pred),3)
        p_srocc = np.round(self.srocc(y_true, y_pred),3)
        p_mae  = np.round(np.mean(np.abs(y_true - y_pred)),3)
        p_rmse  = np.round(np.sqrt(np.mean((y_true - y_pred)**2)),3)
        
        if show_plot:
            print('SRCC: {} | PLCC: {} | MAE: {} | RMSE: {}'.\
                format(p_srocc, p_plcc, p_mae, p_rmse))

            scatter2d = np.vstack((y_true, y_pred)).T
            # Logger.current_logger().report_scatter2d(
            #     title="Correlation",
            #     series="predict correlation",
            #     iteration=0,
            #     scatter=scatter2d,
            #     xaxis="ground-truth",
            #     yaxis="predicted",
            #     mode="markers"
            # )
        self.srocc_score = p_srocc
        self.plcc_score = p_plcc
        self.mae = p_mae
        self.rmse = p_rmse
        return (p_srocc, p_plcc, p_mae, p_rmse)

    def train_model(self, num_epochs=40):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        since = time.time()

        val_plcc_history = []
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_plcc = -float('inf')

        loss_fn = torch.nn.MSELoss()

        for epoch in tqdm(range(num_epochs), total=num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train','val']:
                if phase == 'train':
                    loader = self._train_loader
                    self.model.train()  # Set model to training mode

                else:
                    loader = self._val_loader
                    self.model.eval()   # Set model to evaluate mode
                    # num_batches = np.int(np.ceil(len(ids_val)/batch_size))

                running_loss = 0.0
                running_plcc = 0.0
                # Iterate over data.
    #             for k in tqdm_notebook(range(0,num_batches)):
                loader_size = len(loader)
                print("Loader size: ", loader_size)
                for inputs, labels in loader:
                    labels = labels.to(self.device)
                    inputs = inputs.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.

                        outputs = self.model(inputs).to(self.device)
                        outputs = outputs.squeeze(1)

                        loss = loss_fn(outputs, labels)
                        if phase=='val':
                            plcc_batch = self.plcc(labels.detach().cpu().numpy(),
                                                   outputs.detach().cpu().numpy())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    if phase=='val':
                        running_plcc += plcc_batch


                if phase == 'train':
                    epoch_loss = running_loss / loader_size
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                    # Logger.current_logger().report_scalar(
                    #     "train", "loss", iteration=epoch, value=epoch_loss
                    # )
                else:
                    epoch_loss = running_loss / loader_size
                    epoch_plcc = running_plcc / loader_size
                    # print(phase, epoch_loss, epoch_plcc)
                    print('{} Loss: {:.4f} Plcc: {:.4f}'.format(phase, epoch_loss,epoch_plcc))
                    # Logger.current_logger().report_scalar(
                    #     phase, "PLCC", iteration=epoch, value=epoch_plcc
                    # )

                # deep copy the model
                if phase == 'val' and epoch_plcc > best_plcc:
                    best_plcc = epoch_plcc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_plcc_history.append(epoch_plcc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val PLCC: {:4f}'.format(best_plcc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, val_plcc_history

    def test_model(self):
        self.model.eval()
        y_pred = []
        y_true = []
        for inputs, labels in self._test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                output = self.model(inputs)
                y_pred.append(output.item())
                y_true.append(labels.item())

        self.max = max(y_pred)
        self.min = min(y_pred)
        return self.rating_metrics(y_true, y_pred)


class model_qa(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(model_qa,self).__init__()
        base_model = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.base= nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),         
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x):
        x = self.base(x)
        x = nn.functional.avg_pool2d(x, x.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__=='__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # task = Task.init(project_name="KonCept", task_name="Original KonCept", reuse_last_task_id=False)
    random.seed(10)
    index = list(range(0,10073))
    train_index = index[0:round(0.7*len(index))]
    val_index = index[round(0.7*len(index)):round(0.8*len(index))]
    test_index = index[round(0.8*len(index)):len(index)]
    print('train:', len(train_index), 
          'val:', len(val_index), 
          'test:', len(test_index))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_ft = model_qa(num_classes=1) 
    model_ft = model_ft.to(device)

    params = {
        "ft1_epochs": 40, 
        "ft1_lr": 1e-4, 
        "ft2_epochs": 20, 
        "ft2_lr":1e-5*5,
        "epochs": 10, 
        "lr":1e-5,
        "batch": 16
        }
    # task.connect(params)

    manager1 = ModelManager(model_ft,
                           batch_size=params["batch"],
                           lr=params["ft1_lr"],
                           train_index=train_index, 
                           val_index=val_index, 
                           test_index=test_index, 
                           device=device)

    

    # optimizer_1 = optim.Adam(model_ft.parameters(), lr=params["ft_lr"])
    model_ft_1, val_plcc_history_1 = manager1.train_model(num_epochs=params["ft1_epochs"])
    torch.save(model_ft_1.state_dict(),'./model_ft_1.pth')
    
    manager2 = ModelManager(model_ft_1,
                           batch_size=params["batch"],
                           lr=params["ft2_lr"],
                           train_index=train_index, 
                           val_index=val_index, 
                           test_index=test_index, 
                           device=device)

    model_ft_2, val_plcc_history_2 = manager2.train_model(num_epochs=params["ft2_epochs"])
    torch.save(model_ft_2.state_dict(),'./model_ft_2.pth')


    manager = ModelManager(model_ft_2, 
                            batch_size=params["batch"],
                            lr=params["lr"],
                            train_index=train_index, 
                            val_index=val_index, 
                            test_index=test_index, 
                            device=device)
    # optimizer_2 = optim.Adam(model_ft_1.parameters(), lr=params["lr"])
    KonCept512, val_plcc_history_2 = manager.train_model(num_epochs=params["epochs"])
    torch.save(KonCept512.state_dict(),'./KonCept512.pth')



    ### Test model on the default test set
    KonCept512 = model_qa(num_classes=1) 
    KonCept512.load_state_dict(torch.load('./KonCept512.pth'))
    KonCept512.eval().to(device)
    
    manager.test_model()
    ckpt = {
        "model": KonCept512.state_dict(),
        "min": manager.min,
        "max": manager.max,
        "PLCC": manager.plcc_score,
        "SROCC": manager.srocc_score,
        "MAE": manager.mae,
        "RMSE": manager.rmse,
    }
    torch.save(ckpt, "./KonCept.pt")
    artifacts = {
        "model_name": "KonCept512",
        "min": manager.min,
        "max": manager.max,
        "PLCC": manager.plcc_score,
        "SROCC": manager.srocc_score,
        "MAE": manager.mae,
        "RMSE": manager.rmse,
    }
    # task.upload_artifact(name="Metrics", artifact_object=artifacts)

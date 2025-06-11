"""
Fourier Neural Operator for 2D problems with minimal essential improvements and early stopping.
Based on the original FNO paper (https://arxiv.org/pdf/2010.08895.pdf) with:
- Xavier initialization
- Basic data normalization
- Early stopping
- Adam optimizer without weight decay
- CSV logging for training metrics and model configuration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import operator
from functools import reduce, partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import csv
from datetime import datetime

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2dPro(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dPro, self).__init__()

        """
        2D Fourier layer with Xavier initialization for complex weights.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # Xavier initialization for complex weights
        self.scale = np.sqrt(2.0 / (in_channels + out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2dPro(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2dPro, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        
        # Input projection with Xavier initialization
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        nn.init.xavier_normal_(self.fc0.weight)

        # Fourier layers
        self.conv0 = SpectralConv2dPro(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2dPro(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2dPro(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2dPro(self.width, self.width, self.modes1, self.modes2)
        
        # Linear layers with Xavier initialization
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        for w in [self.w0, self.w1, self.w2, self.w3]:
            nn.init.xavier_normal_(w.weight)

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class EarlyStopping:
    def __init__(self, model, timestamp, patience=10, min_delta=1e-4):
        self.model = model
        self.timestamp = timestamp
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, ep, csv_filename):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(ep, val_loss, csv_filename)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(ep, val_loss, csv_filename)
            self.counter = 0

    def save_checkpoint(self, ep, val_loss, csv_filename):
        model_save_path = f'best_model_fno2d_pro_earlystop_{self.timestamp}.pth'
        torch.save(self.model.state_dict(), model_save_path)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Best model saved at epoch', ep, 'with test_l2', f'{val_loss:.6e}'])

def train_fno2d_pro_earlystop(model, train_loader, test_loader, ntrain, ntest, epochs, learning_rate, device, batch_size, s, patience, min_delta):
    optimizer = Adam(model.parameters(), lr=learning_rate)  # No weight decay
    criterion = nn.MSELoss()
    myloss = LpLoss(size_average=False)
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'training_metrics_2d_pro_earlystop_{timestamp}.csv'
    
    early_stopping = EarlyStopping(model, timestamp, patience=patience, min_delta=min_delta)

    # Write header to CSV
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'time', 'train_mse', 'train_l2', 'test_l2', 'learning_rate'])
    
    # Basic data normalization
    x_train_tensor = train_loader.dataset.tensors[0]
    y_train_tensor = train_loader.dataset.tensors[1]
    
    x_mean = x_train_tensor.mean()
    x_std = x_train_tensor.std()
    y_mean = y_train_tensor.mean()
    y_std = y_train_tensor.std()
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Basic normalization
            x = (x - x_mean) / (x_std + 1e-8)
            
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            
            # Denormalize output
            out_denorm = out * y_std + y_mean
            
            # Calculate losses
            mse = criterion(out_denorm.view(x.shape[0], -1), y.view(x.shape[0], -1))
            l2 = myloss(out_denorm.view(x.shape[0],-1), y.view(x.shape[0],-1))
            
            mse.backward()
            optimizer.step()
            
            train_mse += mse.item()
            train_l2 += l2.item()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                x = (x - x_mean) / (x_std + 1e-8)
                out = model(x).squeeze(-1)
                out_denorm = out * y_std + y_mean

                test_l2 += myloss(out_denorm.view(x.shape[0],-1), y.view(x.shape[0],-1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        epoch_time = t2 - t1
        
        # Print to console
        print(f'{ep}\t{epoch_time:.2f}\t{train_mse:.6e}\t{train_l2:.6e}\t{test_l2:.6e}')
        
        # Write to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, f'{epoch_time:.2f}', f'{train_mse:.6e}', 
                           f'{train_l2:.6e}', f'{test_l2:.6e}', f'{current_lr:.6e}'])
            
        # Early stopping check
        early_stopping(test_l2, ep, csv_filename)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Early stopping triggered at epoch', ep])
            break

if __name__ == "__main__":
    # Configuration
    TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
    TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

    ntrain = 1000
    ntest = 100

    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    patience = 20
    min_delta = 1e-6
    
    modes = 12
    width = 32

    r = 5
    h = int(((421 - 1)/r) + 1)
    s = h

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    x_train = x_train.reshape(ntrain,s,s,1)
    x_test = x_test.reshape(ntest,s,s,1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Initialize and train model
    model = FNO2dPro(modes, modes, width).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Write model configuration to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f'model_config_2d_pro_earlystop_{timestamp}.csv'
    with open(config_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['ntrain', ntrain])
        writer.writerow(['ntest', ntest])
        writer.writerow(['r_subsampling', r])
        writer.writerow(['h_resolution', h])
        writer.writerow(['s_size', s])
        writer.writerow(['batch_size', batch_size])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['epochs', epochs])
        writer.writerow(['patience', patience])
        writer.writerow(['min_delta', min_delta])
        writer.writerow(['modes', modes])
        writer.writerow(['width', width])
        writer.writerow(['total_parameters', sum(p.numel() for p in model.parameters())])
    
    train_fno2d_pro_earlystop(model, train_loader, test_loader, ntrain, ntest, epochs, learning_rate, device, batch_size, s, patience, min_delta) 
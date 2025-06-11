"""
Fourier Neural Operator for 1D problems with minimal essential improvements.
Based on the original FNO paper (https://arxiv.org/pdf/2010.08895.pdf) with only:
- Xavier initialization
- Basic data normalization
No early stopping, fixed 500 epochs.
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

class SpectralConv1dPro(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1dPro, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        
        # Xavier initialization for complex weights
        self.scale = np.sqrt(2.0 / (in_channels + out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1dPro(nn.Module):
    def __init__(self, modes, width):
        super(FNO1dPro, self).__init__()
        
        self.modes1 = modes
        self.width = width
        self.padding = 2
        
        # Input projection with Xavier initialization
        self.fc0 = nn.Linear(2, self.width)
        nn.init.xavier_normal_(self.fc0.weight)
        
        # Fourier layers
        self.conv0 = SpectralConv1dPro(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1dPro(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1dPro(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1dPro(self.width, self.width, self.modes1)
        
        # Linear layers with Xavier initialization
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
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
        x = x.permute(0, 2, 1)
        
        # Fourier layers with residual connections
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
        
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

def train_fno1d_pro(model, train_loader, test_loader, epochs, learning_rate, device):
    optimizer = Adam(model.parameters(), lr=learning_rate)  # No weight decay
    criterion = nn.MSELoss()  # Simple MSE loss
    myloss = LpLoss(size_average=False)
    
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'training_metrics_no_earlystop_{timestamp}.csv'
    
    # Write header to CSV
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'time', 'train_mse', 'train_l2', 'test_l2', 'learning_rate'])
    
    # Basic data normalization
    x_mean = torch.mean(train_loader.dataset.tensors[0])
    x_std = torch.std(train_loader.dataset.tensors[0])
    y_mean = torch.mean(train_loader.dataset.tensors[1])
    y_std = torch.std(train_loader.dataset.tensors[1])
    
    best_test_loss = float('inf')
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Ensure correct shapes
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            if len(y.shape) == 2:
                y = y.unsqueeze(-1)
            
            # Basic normalization
            x = (x - x_mean) / (x_std + 1e-8)
            
            optimizer.zero_grad()
            out = model(x)
            
            # Denormalize output
            out = out * y_std + y_mean
            
            # Calculate losses
            mse = criterion(out.view(batch_size, -1), y.view(batch_size, -1))
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            
            mse.backward()
            optimizer.step()
            
            train_mse += mse.item()
            train_l2 += l2.item()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation
        model.eval()
        test_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)
                if len(y.shape) == 2:
                    y = y.unsqueeze(-1)
                
                x = (x - x_mean) / (x_std + 1e-8)
                out = model(x)
                out = out * y_std + y_mean
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
        
        t2 = default_timer()
        epoch_time = t2 - t1
        
        # Print to console
        print(f'{ep}\t{epoch_time:.2f}\t{train_mse:.6f}\t{train_l2:.6f}\t{test_l2:.6f}')
        
        # Write to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, f'{epoch_time:.2f}', f'{train_mse:.6f}', 
                           f'{train_l2:.6f}', f'{test_l2:.6f}', f'{current_lr:.6f}'])
            
        # Save best model
        if test_l2 < best_test_loss:
            best_test_loss = test_l2
            model_save_path = f'best_model_fno1d_pro_no_earlystop_{timestamp}.pth'
            torch.save(model.state_dict(), model_save_path)
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Best model saved at epoch', ep, 'with test_l2', f'{test_l2:.6f}'])

if __name__ == "__main__":
    # Configuration
    ntrain = 1000
    ntest = 100
    sub = 2**3
    h = 2**13 // sub
    s = h
    batch_size = 20
    learning_rate = 0.001
    epochs = 500  # Fixed 500 epochs
    modes = 16
    width = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]
    
    # Ensure correct shapes for input and output
    x_train = x_data[:ntrain,:].reshape(ntrain,s,1)
    y_train = y_data[:ntrain,:].reshape(ntrain,s,1)
    x_test = x_data[-ntest:,:].reshape(ntest,s,1)
    y_test = y_data[-ntest:,:].reshape(ntest,s,1)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize and train model
    model = FNO1dPro(modes, width).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Write model configuration to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f'model_config_no_earlystop_{timestamp}.csv'
    with open(config_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['ntrain', ntrain])
        writer.writerow(['ntest', ntest])
        writer.writerow(['sub', sub])
        writer.writerow(['h', h])
        writer.writerow(['s', s])
        writer.writerow(['batch_size', batch_size])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['epochs', epochs])
        writer.writerow(['modes', modes])
        writer.writerow(['width', width])
        writer.writerow(['total_parameters', sum(p.numel() for p in model.parameters())])
    
    train_fno1d_pro(model, train_loader, test_loader, epochs, learning_rate, device) 
"""
Enhanced Fourier Neural Operator for 1D problems with improved training stability and performance.
Based on the original FNO paper (https://arxiv.org/pdf/2010.08895.pdf) with additional improvements:
- Xavier/Glorot weight initialization
- Data normalization
- Advanced loss functions (L2 + H1 seminorm + TV regularization)
- Layer normalization and dropout
- Data augmentation with noise injection
- Gradient clipping
- Cosine annealing with warm restarts
- Early stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_
import operator
from functools import reduce, partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv1dPro(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, dropout_rate=0.1):
        super(SpectralConv1dPro, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        
        # Xavier/Glorot initialization
        self.scale = np.sqrt(2.0 / (in_channels + out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        
        # Add layer normalization and dropout
        self.layer_norm = nn.LayerNorm([out_channels, modes1])
        self.dropout = nn.Dropout(dropout_rate)

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes with dropout
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.dropout(
            self.layer_norm(
                self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
            )
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1dPro(nn.Module):
    def __init__(self, modes, width, dropout_rate=0.1):
        super(FNO1dPro, self).__init__()
        
        self.modes1 = modes
        self.width = width
        self.padding = 2
        
        # Input projection with Xavier initialization
        self.fc0 = nn.Linear(2, self.width)
        nn.init.xavier_normal_(self.fc0.weight)
        
        # Fourier layers with dropout
        self.conv0 = SpectralConv1dPro(self.width, self.width, self.modes1, dropout_rate)
        self.conv1 = SpectralConv1dPro(self.width, self.width, self.modes1, dropout_rate)
        self.conv2 = SpectralConv1dPro(self.width, self.width, self.modes1, dropout_rate)
        self.conv3 = SpectralConv1dPro(self.width, self.width, self.modes1, dropout_rate)
        
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
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm([self.width, 1])
        self.layer_norm2 = nn.LayerNorm([self.width, 1])
        self.layer_norm3 = nn.LayerNorm([self.width, 1])
        self.layer_norm4 = nn.LayerNorm([self.width, 1])

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Fourier layers with residual connections and normalization
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.layer_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.layer_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.layer_norm3(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = self.layer_norm4(x.permute(0, 2, 1)).permute(0, 2, 1)
        
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

class EnhancedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super(EnhancedLoss, self).__init__()
        self.alpha = alpha  # Weight for H1 seminorm
        self.beta = beta    # Weight for TV regularization
        
    def forward(self, pred, target):
        # L2 loss
        l2_loss = F.mse_loss(pred, target)
        
        # H1 seminorm (gradient penalty)
        pred_grad = torch.gradient(pred, dim=1)[0]
        target_grad = torch.gradient(target, dim=1)[0]
        h1_loss = F.mse_loss(pred_grad, target_grad)
        
        # Total variation regularization
        tv_loss = torch.mean(torch.abs(torch.diff(pred, dim=1)))
        
        return l2_loss + self.alpha * h1_loss + self.beta * tv_loss

class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_fno1d_pro(model, train_loader, test_loader, epochs, learning_rate, device):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = EnhancedLoss()
    early_stopping = EarlyStopping(patience=50)
    
    # Data normalization
    x_mean = torch.mean(train_loader.dataset.tensors[0])
    x_std = torch.std(train_loader.dataset.tensors[0])
    y_mean = torch.mean(train_loader.dataset.tensors[1])
    y_std = torch.std(train_loader.dataset.tensors[1])
    
    best_test_loss = float('inf')
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Data augmentation with noise injection
            if torch.rand(1) < 0.3:  # 30% chance of noise injection
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            # Normalize input
            x = (x - x_mean) / x_std
            
            optimizer.zero_grad()
            out = model(x)
            
            # Denormalize output for loss calculation
            out = out * y_std + y_mean
            
            loss = criterion(out, y)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                x = (x - x_mean) / x_std
                out = model(x)
                out = out * y_std + y_mean
                test_loss += criterion(out, y).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        t2 = default_timer()
        print(f'Epoch {ep}: Time = {t2-t1:.2f}s, Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}')
        
        # Early stopping check
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model_fno1d_pro.pth')

if __name__ == "__main__":
    # Configuration
    ntrain = 1000
    ntest = 100
    sub = 2**3
    h = 2**13 // sub
    s = h
    batch_size = 20
    learning_rate = 0.001
    epochs = 500
    modes = 16
    width = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]
    
    x_train = x_data[:ntrain,:].reshape(ntrain,s,1)
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:].reshape(ntest,s,1)
    y_test = y_data[-ntest:,:]
    
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
    
    train_fno1d_pro(model, train_loader, test_loader, epochs, learning_rate, device) 
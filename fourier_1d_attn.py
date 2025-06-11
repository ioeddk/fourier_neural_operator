"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
import argparse
import csv
from datetime import datetime
import math
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

class AttentionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(AttentionConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        
        # Instead of weights, we'll use attention
        self.attention = ComplexAttention(in_channels, 1)
        
        # Projection layer to get to output channels
        self.proj = nn.Linear(in_channels, out_channels, dtype=torch.cfloat)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Reshape for attention
        # We'll treat the frequency components as a sequence
        x_ft_reshaped = x_ft.reshape(batchsize, self.in_channels, -1)
        x_ft_reshaped = x_ft_reshaped.permute(0, 2, 1)  # [batch, seq_len, channels]
        
        # Apply self-attention
        attn_output = self.attention(x_ft_reshaped)
        
        # Project to output channels
        attn_output = self.proj(attn_output)
        
        # Reshape back to original format
        out_ft = attn_output.permute(0, 2, 1).reshape(
            batchsize, self.out_channels, x.size(-1)//2 + 1
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft)
        return x
class ComplexAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ComplexAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = 2 * embed_dim // num_heads # Updated to account for doubled dimension
        
        # Complex-valued projections
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim, dtype=torch.cfloat)
        self.k_proj = nn.Linear(embed_dim, 2 * embed_dim, dtype=torch.cfloat)
        self.v_proj = nn.Linear(embed_dim, 2 * embed_dim, dtype=torch.cfloat)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim, dtype=torch.cfloat)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention with doubled dimension
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # >>> saparate real and imaginary parts softmax >>>
        # # Separate real and imaginary parts and apply softmax separately
        # real_scores = scores.real
        # imag_scores = scores.imag
        
        # # Apply softmax to real and imaginary parts
        # real_scores = F.softmax(real_scores, dim=-1)
        # imag_scores = F.softmax(imag_scores, dim=-1)
        
        # # Recombine into complex tensor
        # normalized_scores = torch.complex(real_scores, imag_scores)
        # <<< saparate real and imaginary parts softmax <<<

        # phase preserving softmax
        probs = F.softmax(torch.abs(scores), dim=-1)
        scores = probs * scores / torch.abs(scores)
        
        # Apply attention
        output = torch.matmul(scores, v)
        output = output.view(batch_size, seq_len, 2 * self.embed_dim) # Updated dimension
        
        # Final projection
        output = self.out_proj(output)
        return output

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, mode='classic'):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        
        # if mode == 'classic':
        #     self.spectral_conv = SpectralConv1d
        # elif mode == 'attn':
        #     self.spectral_conv = AttentionConv1d
        # else:
        #     raise ValueError(f"Invalid mode: {mode}")

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.attn0 = AttentionConv1d(self.width, self.width, self.modes1)
        self.attn1 = AttentionConv1d(self.width, self.width, self.modes1)
        self.attn2 = AttentionConv1d(self.width, self.width, self.modes1)
        self.attn3 = AttentionConv1d(self.width, self.width, self.modes1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.attn0(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.attn1(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.attn2(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.attn3(x)
        x = x1 + x2 + x3

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
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

################################################################
#  configurations
################################################################

# Take in some command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='data/burgers_data_R10.mat', help='path to training data')
parser.add_argument('--test_path', type=str, default='data/burgers_data_R10.mat', help='path to test data')
parser.add_argument('--ntrain', type=int, default=1000, help='number of training samples')
parser.add_argument('--ntest', type=int, default=100, help='number of test samples')
parser.add_argument('--batch_size', type=int, default=20, help='batch size for training')
parser.add_argument('--output_dir', type=str, default='output', help='output directory')
args = parser.parse_args()

TRAIN_PATH = args.train_path
TEST_PATH = args.test_path
ntrain = args.ntrain
ntest = args.ntest

# Create output directory structure
output_dir = args.output_dir
run_name = f'burgers_fourier_N{ntrain}_Ntest{ntest}'

os.makedirs(output_dir, exist_ok=True)
# Create files for train and test errors
train_err_file = os.path.join(output_dir, f'{run_name}_train_error.txt')
test_err_file = os.path.join(output_dir, f'{run_name}_test_error.txt')

# Open and create the files (this will create empty files)
open(train_err_file, 'w').close()
open(test_err_file, 'w').close()


# Create files for train and test errors
train_err_file = os.path.join(output_dir, f'{run_name}_train_error.txt')
test_err_file = os.path.join(output_dir, f'{run_name}_test_error.txt')

# Open and create the files (this will create empty files)
open(train_err_file, 'w').close()
open(test_err_file, 'w').close()

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = args.batch_size
learning_rate = 0.0001
epochs = 500
step_size = 50
gamma = 0.5

modes = 16
width = 64

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader(TRAIN_PATH)
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width, mode='attn').cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Create CSV file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'training_metrics_1d_{timestamp}.csv'

# Write header to CSV
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'time', 'train_mse', 'train_l2', 'test_l2', 'learning_rate'])

best_test_loss = float('inf')

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    epoch_time = t2 - t1
    print(ep, epoch_time, train_mse, train_l2, test_l2)
    with open(train_err_file, 'a') as f:
        f.write(f'{ep} {epoch_time} {train_mse} {train_l2} {test_l2}\n')
    
    # Write to CSV
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ep, f'{epoch_time:.2f}', f'{train_mse:.6f}', 
                        f'{train_l2:.6f}', f'{test_l2:.6f}', f'{current_lr:.6f}'])
    
    # Save best model
    if test_l2 < best_test_loss:
        best_test_loss = test_l2
        model_save_path = f'best_model_fno1d_{timestamp}.pth'
        torch.save(model.state_dict(), model_save_path)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Best model saved at epoch', ep, 'with test_l2', f'{test_l2:.6f}'])

pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).view(-1)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        with open(train_err_file, 'a') as f:
            f.write(f'{index} {test_l2}\n')
        with open(test_err_file, 'a') as f:
            f.write(f'{index} {test_l2}\n')
        index = index + 1

# Save model
model_path = os.path.join(output_dir, 'models', run_name + '.pt')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)

# Save predictions
pred_path = os.path.join(output_dir, 'pred', run_name + '.mat')
os.makedirs(os.path.dirname(pred_path), exist_ok=True)
scipy.io.savemat(pred_path, mdict={'pred': pred.cpu().numpy()})

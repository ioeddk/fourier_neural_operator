# Enhanced Fourier Neural Operator

This repository contains enhanced implementations of the Fourier Neural Operator with additional improvements for better training stability and performance. 

You need to download the dataset from the [original author's repo](https://github.com/wesley-stone/fourier_neural_operator/tree/master), [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) specifically, then change the path pointing to those datasets in the corresponding python script. The dataset used for 2D problems is from [Kaggle](https://www.kaggle.com/datasets/sentinelprimehk/darcy-flow-equation). And for the following tasks, use the command below to run: 

## 1D
- 1D Baseline: `python fourier_1d.py`
- 1D with Improvements 1: `python fourier_1d_pro.py`
- 1D with Improvements 2: `python fourier1dpro_no_earlystop.py`
- 1D with Fourier Attention: `./run_fourier_1d_attn.sh`

## 2D
- 2D Baseline: `python fourier_2d.py`
- 2D with Improvements 1: `python fourier_2d_pro.py`
- 2D with Improvements 2: `python fourier_2d_pro_earlystop.py`

## 3D
- 3D Baseline: `python fourier_3d.py`
- 3D with Improvements 1: `python fourier_3d_pro.py`
- 3D with Improvements 2: `python fourier_3d_pro_earlystop.py`

The plots are created with `plot.ipynb` and `plot1.ipynb`. 
Notice, for the command involing the bash script, you need to change the name to your own conda environment in the script first. 

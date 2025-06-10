#!/bin/bash


eval "$(conda shell.bash hook)"
conda activate ece228

        # {
        #     "name": "Run: fourier_3d_pro_no_earlystop.py",
        #     "type": "python",
        #     "request": "launch",
        #     "program": "${workspaceFolder}/fourier_3d_pro_no_earlystop.py",
        #     "console": "integratedTerminal",
        #     "justMyCode": false,
        #     "purpose": ["debug-in-terminal"],
        #     "args": [
        #         "--train_path", "/home/ubuntu/ece228/fourier_neural_operator/data/ns_V1e-3_N5000_T50_tiny.mat",
        #         "--test_path", "/home/ubuntu/ece228/fourier_neural_operator/data/ns_V1e-3_N5000_T50_tiny.mat",
        #         "--ntrain", "500",
        #         "--ntest", "100",
        #         "--output_dir", "output_3d_no_earlystop",
        #         "--batch_size", "10"
        #     ],
        #     "env": {
        #         "PYTHONPATH": "${workspaceFolder}"
        #     }
        # },

python fourier_3d_pro.py --train_path=/home/ubuntu/ece228/fourier_neural_operator/data/ns_V1e-3_N5000_T50_tiny.mat --test_path=/home/ubuntu/ece228/fourier_neural_operator/data/ns_V1e-3_N5000_T50_tiny.mat --ntrain=500 --ntest=100 --output_dir=output_3d --batch_size=10
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ece228


        # {
        #     "name": "Run: fourier_1d_attn.py",
        #     "type": "python",
        #     "request": "launch",
        #     "program": "${workspaceFolder}/fourier_1d_attn.py",
        #     "console": "integratedTerminal",
        #     "justMyCode": false,
        #     "purpose": ["debug-in-terminal"],
        #     "args": [
        #         "--train_path", "data/burgers_data_R10.mat",
        #         "--test_path", "data/burgers_data_R10.mat",
        #         "--ntrain", "1000",
        #         "--ntest", "100",
        #         "--output_dir", "output",
        #         "--batch_size", "20"
        #     ],
        #     "env": {
        #         "PYTHONPATH": "${workspaceFolder}"
        #     }
        # },

python fourier_1d_attn.py --train_path data/burgers_data_R10.mat --test_path data/burgers_data_R10.mat --ntrain 1000 --ntest 100 --output_dir output_1d_attn --batch_size 20
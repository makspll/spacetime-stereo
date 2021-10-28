#!/bin/bash

eval "$(conda shell.bash hook)"


conda deactivate

# conda will ask to remove old ones with same name if they exist

# LEAStereo
conda create -n leastereo python=3.8 
conda activate leastereo
conda install pip pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 "matplotlib<3.3" path.py tensorboard tensorboardX tqdm scipy scikit-image opencv -c pytorch -c conda-forge
pip install torchsummary
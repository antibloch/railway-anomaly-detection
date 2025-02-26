#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropiate environment

conda create -n railanomaly python=3.8 -y && conda activate railanomaly

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install --no-dependencies ultralytics


pip install numpy pandas opencv-python matplotlib seaborn psutil PyYAML tqdm


pip install scikit-learn

conda install scikit-image -y


pip install pyyaml

pip install python-xml2dict

pip install h5py


pip install transformers accelerate

pip install colour

pip install rich

pip install timm opacus

pip install kagglehub

pip install kaggle

pip install tensorboard

pip install torchmetrics
pip install torchgeometry 
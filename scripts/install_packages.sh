#!/bin/bash

echo "****** create mmsampler conda env ******"
conda create -n mmsampler python=3.7 -y
conda activate mmsampler

echo "****** install pytorch torchvision cudatoolkit ******"
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0

echo "****** install other libs ******"
pip install tqdm tensorboard easydict ftfy regex ptflops efficientnet_pytorch pandas mlflow

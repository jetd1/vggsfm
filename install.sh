# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.10, CUDA 12.1

conda deactivate

# Set environment variables
export ENV_NAME=vggsfm
export PYTHON_VERSION=3.10
export PYTORCH_VERSION=2.1.0
export CUDA_VERSION=12.1

# Create a new conda environment and activate it
conda create -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install PyTorch, torchvision, and PyTorch3D using conda
conda install pytorch=$PYTORCH_VERSION torchvision pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install pip packages
pip install hydra-core --upgrade
pip install omegaconf opencv-python einops visdom 
pip install accelerate==0.24.0

pip install gluefactory
# Ensure the version of pycolmap is 0.5.0
pip install pycolmap==0.5.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This Script Assumes Python 3.10, CUDA 12.1, pytorch 2.3.0

conda deactivate

# Set environment variables
export ENV_NAME=vggsfm

# Create a new conda environment and activate it
conda create -n $ENV_NAME -y python=3.10 && \
conda activate $ENV_NAME && \
python -m pip install torch==2.3.0 torchvision==0.18.0 fvcore==0.1.5.post20221221 iopath==0.1.9 hydra-core==1.3.2 omegaconf==2.3.0 opencv-python==4.9.0.80 einops==0.8.0 visdom==0.2.4 accelerate==0.24.0 pycolmap==0.5.0 && \
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu121 && \
python -m pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6 git+https://github.com/cvg/glue-factory.git@1f56839db2242929960d70f85bfac6c19ef2821c
#!/bin/bash
conda env create -f environment.yml
conda activate alan
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
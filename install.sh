#!/usr/bin/env bash

echo "Step 1: install miniconda (if not present)"
if [ ! -d ~/miniconda3 ]; then
  echo "Installing Miniconda"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || exit
  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
fi

echo "Step 2: activate conda and install the environment with the name 'DecomposeTransformer'"
# shellcheck disable=SC1090

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env create -f environment.yml -y
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate DecomposeTransformer
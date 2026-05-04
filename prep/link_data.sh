#!/bin/bash

DATASET=$1
ROOT_DIR=$(pwd)

# Symlink the dataset directory
ln -s "$ROOT_DIR/data/$DATASET" external/motion-diffusion-model/dataset
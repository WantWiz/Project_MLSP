#!/bin/bash

set -e

git submodule update --init --recursive

pip install -U demucs

# install encodecmae
cd encodecmae
pip install -e .
cd ../

# install encodecmae-to-wav
cd encodecmae-to-wav
pip install -e .
cd ../

# install pecmae
pip install -e .

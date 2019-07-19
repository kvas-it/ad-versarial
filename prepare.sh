#!/bin/bash

# I tested it with python3.5.7, other versions might work too.
PYTHON=python3.5

if [ -d venv ]; then
    echo "## Virtualenv already exists"
else
    echo "## Creating the virtualenv"
    $PYTHON -m venv venv
    echo "## Installing dependencies"
    venv/bin/pip install -r requirements.txt
fi

mkdir -p tmp

DATA_TGZ=tmp/data.tar.gz
DATA_URL=https://github.com/ftramer/ad-versarial/releases/download/0.1/data.tar.gz

if [ -e data/page_based ]; then
    echo "## Data is already in place"
else
    if [ -f $DATA_TGZ ]; then
        echo "## Data already downloaded -- will use that"
    else
        echo "## Downloading data from $DATA_URL"
        curl -L -o $DATA_TGZ $DATA_URL
    fi
    echo "## Extracting data"
    tar xvzf $DATA_TGZ
fi

MODELS_TGZ=tmp/models.tar.gz
MODELS_URL=https://github.com/ftramer/ad-versarial/releases/download/0.1/models.tar.gz

if [ -f models/page_based_yolov3.weights ]; then
    echo "## Model weights are already in place"
else
    if [ -f $MODELS_TGZ ]; then
        echo "## Models already downloaded -- will use that"
    else
        echo "## Downloading models from $MODELS_URL"
        curl -L -o $MODELS_TGZ $MODELS_URL
    fi
    echo "## Extracting model weights"
    tar xvzf $MODELS_TGZ
fi

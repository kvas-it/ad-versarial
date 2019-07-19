#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $0 SOURCE-DIR"
    exit 1
else
    SRC_DIR=`realpath "$1"`
fi

if [ -d output ]; then
    echo "## Removing previous output in output/"
    rm -Rf output
fi

source venv/bin/activate
cd page-based

echo "## Running page-based/classify.py on $SRC_DIR"
python classify.py "--input_dir=$SRC_DIR" --output_dir ../output

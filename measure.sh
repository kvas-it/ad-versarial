#!/bin/bash

if [ "$1" == "" ]; then
    echo "Usage: $0 SOURCE-DIR [FLAGS]"
    exit 1
else
    SRC_DIR=`realpath "$1"`
    shift
fi

if [ -d output ]; then
    echo "## Removing previous output in output/"
    rm -Rf output
fi

source venv/bin/activate
cd page-based

echo "## Running page-based/measure.py on $SRC_DIR"
python measure.py $* "--input_dir=$SRC_DIR" --output_dir ../output

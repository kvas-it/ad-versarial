#!/bin/bash

source venv/bin/activate
cd page-based

echo "## Running page-based/serve.py $*"
python serve.py $*

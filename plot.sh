#!/bin/bash

echo " "
echo "Job started at `date`"

python3 python_code/plot_data.py "$@"

echo " "
echo "Job ended at `date`"
echo " "

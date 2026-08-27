#!/bin/bash

echo " "
echo "Job started at `date`"

python3 fhl-home/for_ssh/python_code/plot_data.py "$@"

echo " "
echo "Job ended at `date`"
echo " "

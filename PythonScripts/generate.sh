#!/bin/bash
python DataGenerator.py -ofile=../Data/data_3d.csv -dim=3d -n=200
python ViewDataset.py -ifile=../Data/data_3d.csv -ofile=../Data/plot_3d.png

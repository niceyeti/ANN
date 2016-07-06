#!/bin/bash
python DataGenerator.py -ofile=../Data/test2d.csv -dim=2d -n=100
python ViewDataset.py -ifile=../Data/test2d.csv -ofile=../Data/test2d.png

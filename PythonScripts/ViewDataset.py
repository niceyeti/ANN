"""
Script for consuming a 2d or 3d training csv data file and displaying it using matplotlib.
The file lines are formatted as: <X,Y,+/-1>
The X and Y are some attribute values, and the +/-1 indicates binary class membership.

"""

from __future__ import print_function
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

"""
Given an input file of 2d, binary class examples, returns a list of (X,Y) tuples
for the selected class ("+" or "-").

@examples: A list of example vectors, formatted as <attr1,attr2...,+/-1>
@binaryClass: A "+" or "-" string indicating which class' examples we wish to read

Returns: A list of (X,Y,class) tuples corresponding with the binaryClass; the class labels are preserved
in the returned list.
"""
def getClassExamples(examples, binaryClass):
	classExamples = []
	for example in examples:
		tokens = example.split(",")
		if tokens[-1] == binaryClass:
			classExamples += (tokens[0],tokens[1],tokens[2])
	return classExamples

"""
Given an inputPath to some csv data file, reads the data into a list of lists of floats and returns this list.
"""
def ReadExamples(inputPath):
	ifile = open(inputPath,"r")
	examples = []
	
	for line in ifile.readlines():
		example = [float(val) for val in line.split(",")]
		examples.append(example)
		
	return examples
	
"""
Takes a list of lists of float vectors and plots them in 2d or 3d, depending on the dimension of the vectors.

@examples: a list of float vectors (lists), formatted as [X,Y,Z,+/-1] or [X,Y,+/-1].
"""
def PlotExamples(examples,outputPath):
	dim = len(examples[0]) - 1

	#configure the plots
	fig = plt.figure(figsize=(14,10))
	if dim == 2:
		ax = fig.add_subplot(111)
	if dim == 3:
		ax = fig.add_subplot(111, projection="3d")	

	#parse the positive and negative examples
	X1 = []
	Y1 = []
	Z1 = []
	X2 = []
	Y2 = []
	Z2 = []
	
	#get the positive examples; only the sign of the class label is used for binary classification
	X1 = [example[0] for example in examples if example[-1] > 0]
	Y1 = [example[1] for example in examples if example[-1] > 0]
	if dim == 3:
		Z1 = [example[2] for example in examples if example[-1] > 0]

	#get the negative examples, again via the negative sign only
	X2 = [example[0] for example in examples if example[-1] < 0]
	Y2 = [example[1] for example in examples if example[-1] < 0]
	if dim == 3:
		Z2 = [example[2] for example in examples if example[-1] < 0]	
	
	#plot the respective class examples
	if dim == 2:
		ax.scatter(X1,Y1, c="b", marker='o')
		ax.scatter(X2,Y2, c="r", marker='x')
	elif dim == 3:
		ax.scatter(X1,Y1, Z1, c="b", marker='o')
		ax.scatter(X2,Y2, Z2, c="r", marker='x')		

	plt.savefig(outputPath)
	plt.show()

	
def usage():
	print("Usage: python ViewDataset.py -ifile=[input csv data file] -ofile=[path to which png will be saved]")
	print("The data must be formatted as csv vectors <X,Y,+/-1> or <X,Y,Z,+/-1>.")
	print("The dimension of the data (2d or 3d) is inferred from the records.")
	print("Also, the positive examples don't need to be labelled '+', just '1'. Either is fine.")
	
if len(sys.argv) != 3:
	print("ERROR wrong number of cmd line args")
	usage()
	exit()
if "-ifile=" not in sys.argv[1]:
	print("ERROR no dataset passed")
	usage()
	exit()
if "-ofile=" not in sys.argv[2]:
	print("ERROR not output path passed")
	usage()
	exit()

ifilePath = sys.argv[1].split("=")[1].strip()
ofilePath = sys.argv[2].split("=")[1].strip()

print("Processing data from: "+ifilePath)
#read the example vectors
examples = ReadExamples(ifilePath)
#plot the examples; the dimension will be inferred from the size of the example vectors
PlotExamples(examples,ofilePath)



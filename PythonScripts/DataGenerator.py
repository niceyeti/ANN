"""
Script for generating 3d or 2d binary class data, +1 and -1. Equal numbers of + and - instances will be created.

The +1 points will be centered about the origin, while the -1 points will be beyond the origin in a shell. For 3d,
the pattern is a Saturn pattern.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sys
import math

"""
Generates data, where the positive examples are centered about the origin, and the
negative examples lie in a shell surrounding the origin. Both sets are generated from some
zero-mean gaussian distribution.

@binaryclass: A "+" or "-" string representing the target class to be generated
@dim: Integer 2 or 3, representing the desired dimension of the data.

Returns: A list of lists of coordinate points. [[x1,y1,z1],[x2,y2,z2], ... [xn,yn,zn]]
If dim==2, then the list is [X,Y]; if dim==3, then the list is [X,Y,Z].
"""
def GenerateData(n, binaryClass,dim=2):
	X = []
	Y = []
	Z = []

	if binaryClass == "+" :
		#generate a bunch of points near the origin
		X = [random.gauss(0,0.5) for i in range(0,n)]
		Y = [random.gauss(0,0.5) for i in range(0,n)]
		Z = [random.gauss(0,0.5) for i in range(0,n)]
	elif binaryClass == "-":
		#good enough approximation of a circle, for now
		radius = 3
		for i in range(0,n):
			theta = float(random.randint(0,90))
			X.append(radius * math.cos(theta) + random.gauss(0.0,0.25))
			Y.append(radius * math.sin(theta) + random.gauss(0.0,0.25))
			Z.append(radius * math.cos(theta) + random.gauss(0.0,0.25))
		
	if dim == 2:
		data = [[x,y] for x,y in zip(X,Y)]
	else:
		data = [[x,y,z] for x,y,z in zip(X,Y,Z)]

	return data

"""
Writes a list of examples to csv. Class membership is not passed, and expected to be the last member of each example


@ofile: an output file handle
@examples: A list of example vectors
@dim: 2 or 3, representing the dimension of the examples
@isTestData: If this is test data, not training data, write without the class labels
"""
def WriteExamples(ofile,examples,dim,isTestData=False):
	for example in examples:
		if not isTestData: #not test data, so output data with class labels
			if dim == 2:
				record = str(example[0])+","+str(example[1])+","+str(example[2])
			elif dim == 3:
				record = str(example[0])+","+str(example[1])+","+str(example[2])+","+str(example[3])
		else:
			if dim == 2:
				record = str(example[0])+","+str(example[1])
			elif dim == 3:
				record = str(example[0])+","+str(example[1])+","+str(example[2])

		ofile.write(record+"\n")
	
def usage():
	print("Usage: python DataGenerator.py -ofile=[path to output file] -dim=[2d or 3d for desired dimensionality] -n=[number of data points to generate] -isTest")
	print("The isTest parameter is optional, but can be passed to indicate the data should be output without class labels.")
	
if len(sys.argv) < 4:
	print("ERROR incorrect number of command line parameters")
	usage()
	exit()
if "-ofile=" not in sys.argv[1]:
	print("ERROR not output file path passed")
	usage()
	exit()
if "-dim=" not in sys.argv[2]:
	print("ERROR no dimension passed")
	usage()
	exit()
if "-n=" not in sys.argv[3]:
	print("ERROR no n param passed")
	usage()
	exit()
isTest = "-isTest" in sys.argv

ofile = open(sys.argv[1].split("=")[1],"w+")
dim = int(sys.argv[2].split("=")[1].strip().replace("d",""))
n = int(sys.argv[3].split("=")[1].strip())

#generate the positive and negative instances
positiveInstances = GenerateData(int(n/2),"+", dim)
#add class labels to the data
for instance in positiveInstances:
	instance.append(1.0)

negativeInstances = GenerateData(int(n/2),"-", dim)
#add class labels to the data
for instance in negativeInstances:
	instance.append(-1.0)

#write out the data; the order is NOT randomized, outputting contiguous blocks of + and - examples
WriteExamples(ofile,positiveInstances,dim,isTest)
WriteExamples(ofile,negativeInstances,dim,isTest)
	
ofile.close()
	
"""
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection="3d")

X = [i for i in range(1,11)]
Y = [(random.randint(1,10)) for i in range(1,11)]
Z = [(2 * i) for i in range(1,11)]

#generate a bunch of points near the origin
X1 = [random.gauss(0,2) for i in range(0,30)]
Y1 = [random.gauss(0,2) for i in range(0,30)]
Z1 = [random.gauss(0,2) for i in range(0,30)]

#generate a bunch of points further from the origin in a shell
X2 = [random.gauss(0,2) * 4 for i in range(0,30)]
Y2 = [random.gauss(0,2) * 4 for i in range(0,30)]
Z2 = [random.gauss(0,2) * 4 for i in range(0,30)]

ax.scatter(X1,Y1,Z1, c="r", marker='o')
ax.scatter(X2,Y2,Z2, c="b", marker='o')

plt.show()
"""
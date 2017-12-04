"""
This is for using synthetic data from my process mining thesis project to test a adjacency-matrix based autoencoder design.

The data in the process mining project consists of collections of graphs, one per each trace generated from a process model (a graph).
The input files are stored off for other purposes in "traceGraph.py" files as adjacency-matrices (one per trace), as edge lists: 

	...
	(1,[('3', 'l'), ('l', 'E'), ('E', 'U'), ('U', 'a'), ('a', 'A')])
	....
	
This will be converted to output like:
	1,0000001010000000100000001000001010100000...
	
Tuples of the form tup[0]=trace-id, and tup[1]=adjacency list.

This conversion script converts traceGraph.py files to a binary string to serve as input to an autoencoder.
An adjacency matrix is size |vertices|**2. The rows of this matrix are concatenated into a single multi-hot vector 
of '1' and '0' characters as input to an autoencoder of some form.

Note the similarity of this input to NIST data... its worth thinking about problem similarity and decomposition.

"""

from __future__ import print_function
import sys
import os
import traceback
import numpy as np
import random

try:
	foo = raw_input
except:
	raw_input = input


def usage():
	print("Usage: python TraceConverter.py --tracePath=[path to trace file] --opath=[path to output file]")

#From the trace file, builds a dict mapping vertex names to row-ids
def _getVertexDict(lines):
	#build the vertex set, as a dictionary mapping string vertex names to 
	vertexDict = dict()
	idCt = 0
	#print("Lines: "+str(len(lines)))
	for line in lines:
		try:
			trace = eval(line)
			id = trace[0]
			random.shuffle(trace[1])
			adjacencyList = trace[1]
			for tup in adjacencyList:
				for vertex in tup:
					if vertex not in vertexDict:
						vertexDict[vertex] = idCt
						idCt += 1
		except:
			traceback.print_exc()

	#print(str(vertexDict))

	return vertexDict

def _adjacencyListToBinaryString(vertexDict, adjacencyList):
	
	adjacencies = np.zeros(shape=(len(vertexDict.keys()),len(vertexDict.keys())), dtype=np.int32)
	
	for edge in adjacencyList:
		row = vertexDict[edge[0]]
		col = vertexDict[edge[1]]
		adjacencies[row,col] = 1
	
	#Create the string of concatenated rows: r1+r2+r3...
	binString = ""
	for row in range(adjacencies.shape[0]):
		rowVec = adjacencies[row,]
		for col in range(rowVec.shape[0]):
			binString += str(int(rowVec[col]))
		#binString += "\n"

	return binString

def Convert(tracePath, opath):
	
	print("Converting input traces...")
	with open(tracePath, "r") as ifile:
		with open(opath, "w+") as ofile:
			with open(opath[0:opath.find(".")]+"_vertexDict.py", "w+") as idFile: #merely for mapping vertices back to their names, if needed
				lines = [line.strip() for line in ifile.readlines() if len(line.strip()) > 0]
				vertexDict = _getVertexDict(lines)
				idFile.write(str(vertexDict))
				#convert each line's adjacency matrix to a binary string representation
				for line in lines:
					#print(line)
					try:
						trace = eval(line)
						id = trace[0]
						adjacencyList = trace[1]
						binString = _adjacencyListToBinaryString(vertexDict, adjacencyList)
						#ofile.write(str(id)+","+binString+"\n")
						ofile.write(binString+"\n")
					except:
						traceback.print_exc()
					#print(binString)


def main():
	if len(sys.argv) < 3:
		usage()
		exit()
		
	tracePath = ""
	opath = ""
	for arg in sys.argv:
		if "--tracePath=" in arg:
			tracePath = arg.split("=")[1]
		if "--opath=" in arg:
			opath = arg.split("=")[1]

	if len(tracePath) == 0 or len(opath) == 0:
		usage()
		exit()
	
	if not os.path.exists(tracePath):
		print("File does not exist: "+tracePath)
		usage()
		exit()

	if os.path.exists(opath):
		response = raw_input("Path "+opath+" exists. Are you sure? (y)  ")
		if response != "y":
			usage()
			exit()
		
	Convert(tracePath, opath)
	
	
	
	
	
	
if __name__ == "__main__":
	main()


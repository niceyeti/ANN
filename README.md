The ANN object Train() takes a training path, with lines formatted as a list of attribute values followed by a 
+/-1 representing binary class membership, as:
	[attribute 1],[attribute 2], ...,+/-1	
The ANN is binary for now. The dimension of the data is inferred from the dataset, so if the training data
is two-dimensional, the ANN will build its internal data structures to accept two-dimensional inputs (plus the bias, of course).
 
The ANN can then output a prediction file for all examples, formatted exactly as the training data, except that each
line is appended with a second +/-1 representing prediction class membership. So a 2d training data file may
look like:
2.3,4.5,-1
And the prediction output file would then look like:
2.3,4.5,-1,1
For this example, since -1 != 1, this represents a misclassified example.

The intent is just to have an output file(s) that can be used consumed by an external program for displaying
training progress and performance visually, such as with python.
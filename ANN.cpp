#include "ANN.hpp"

MultiLayerNetwork::MultiLayerNetwork(int numInputs, int numHiddenUnits, int numOutputUnits)
{
	_buildTwoLayerNet(numInputs,numHiddenUnits, numOutputUnits);
}

MultiLayerNetwork::~MultiLayerNetwork()
{
	_layers.clear();
}

void MultiLayerNetwork::Clear()
{
	_layers.clear();
	_biases.clear();
}

//Just a utility for nullifying the input ptrs of some layer
void MultiLayerNetwork::_nullifyLayer(vector<Neuron>& layer)
{
	int i, j;
	
	for(i = 0; i < layer.size(); i++){
		for(j = 0; j < layer[i].Inputs.size(); j++){
			layer[i].Inputs[j] = NULL;
		}
	}
	
}

/*
A builder; this could be moved to some builder class

@dataDimension: The dimension of the input data, such as 2 or 3-d. The network neurons will be configured with dataDimension + 1 inputs,
where the additional input is for the bias.
@numHiddenUnits: Number of hidden units in the hidden layer
@numOutputUnits: Nmber of output units
*/
void MultiLayerNetwork::_buildTwoLayerNet(int dataDimension, int numHiddenUnits, int numOutputUnits)
{
	int i, j;
	
	//clear any existing model
	Clear();
	
	//just two layers for now
	_layers.resize(2);
	
	//set up the hidden units
	_layers[0].resize(numHiddenUnits, Neuron(dataDimension + 1, TANH) );  //plus one for the biases
	_nullifyLayer(_layers[0]);
	
	//set up the output units, rigging their inputs to the outputs of the previous hidden layer
	_layers[_layers.size()-1].resize(numOutputUnits, Neuron(numHiddenUnits + 1, TANH) ); //plus one for the biases
	_nullifyLayer(_layers[1]);
	
	vector<Neuron>& outputLayer = _layers[_layers.size() - 1];
	vector<Neuron>& hiddenLayer = _layers[_layers.size() - 2];
	
	//connect the two layers
	for(i = 0; i < outputLayer.size(); i++){
		for(j = 0; j < hiddenLayer.size(); j++){
			//assign the (j+1)th previous neuron's output to the ith neuron's jth input; the +1 is to account for the bias
			outputLayer[i].Inputs[j+1] = &hiddenLayer[j].Output;
		}
	}

	//connect the bias inputs; the zeroeth input of every neuron will be a bias weight
	_biases.resize(2);
	//set the first layer bias ptrs
	for(i = 0; i < hiddenLayer.size(); i++){
		hiddenLayer[i].Inputs[0] = &_biases[0];
	}
	//set the final layer bias ptrs
	for(i = 0; i < outputLayer.size(); i++){
		outputLayer[i].Inputs[0] = &_biases[1];
	}
	//initialize the biases to constant 1.0
	for(i = 0; i < _biases.size(); i++){
		_biases[i] = 1.0;
	}
}

/*
Prints weights of the entire network.
*/
void MultiLayerNetwork::PrintWeights()
{
	int i, j, l;
	
	for(l =0; l < _layers.size(); l++){
		cout << "Layer " << l << " weights: " << endl;
		for(i = 0; i < _layers[l].size(); i++){
			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				//cout << _layers[l][i].Weights[j] << " ";
				printf("%f ",_layers[l][i].Weights[j]);
			}
			cout << endl;
		}
	}
}


/*
There are specific strategies for initializing the weights (see Haykin). Here they are just init'ed with random numbers.
*/
void MultiLayerNetwork::_assignRandomWeights()
{
	int i, j, l;
	
	srand(time(NULL));
	
	//TODO: change the assignment to a zero-mean Gaussian, or other init per lit recommendations.
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				_layers[l][i].Weights[j] = ((double)(rand() % 100)) / 50.0;
				if(rand() % 2 == 0){
					_layers[l][i].Weights[j] *= -1; //flip the sign 50% of the time
				}
			}
		}
	}
	
	//init the biases as well
	for(i = 0; i < _biases.size(); i++){
		_biases[i] = 1.0;
	}
}



/*
Runs the backpropagation algorithm from Duda. Also see Mustafa's Neural Nets CalTech lecture,
which this mirrors.

Another very terse source: http://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html

The gist:
	Initialization:
		Initialize the weights to random/Gaussian values with zero mean and variance such that the standard deviation lies
		at the transition between the linear and saturated parts of the activation function (sigmoid or tanh).

	Algorithm:
		Iterate over the examples e, doing the following:
		Forward compute the activation for all neurons using the e
		-Calculate the error at the output nodes: delta_o = (t_k - z_k) * phi_prime(signal)
		-Calculate the error at the hidden layer: dW = phi_prime(signal) * Sigma( delta_o * w_io )
		-Update the weights for the hidden layer, using these dW's
		-Update the weights for the final layer, using delta
		-Iterate, until the error for the network falls below some threshold. The most straightforward error is simply the sum-of-squares of the delta_o

	Stopping criteria: 
		Ad hoc for now. Can use maxIterations, error-reduction-threshold, or otherwise.
		
*/
void MultiLayerNetwork::BackPropagation(const vector<vector<double> >& dataset)
{
	string dummy;
	int i, j, l, iterations, ringIndex;
	double convergence, input, netError, prevNetError, sum, dW;
	vector<double> errorHistory;
	
	//intialize the weights to random values
	_assignRandomWeights();
	
	//sliding error-window for viewing long term error changes
	errorHistory.resize(50, 0.0);
	ringIndex = 0;
	
	iterations = 0;
	netError = 1;
	convergence = 0;
	_eta = 0.1;
	//while(netError > convergenceThreshold){
	while(iterations < 100000){
		//PrintWeights();
		
		//randomly choose an example
		const vector<double>& example = dataset[ rand() % dataset.size() ];
		//Run the forward pass of the network to set all neuron outputs for this sample
		Classify(example);
		
		//calculate the final-layer deltas, which are just the signal-prime values
		vector<Neuron>& finalLayer = _layers[_layers.size()-1];
		for(i = 0; i < finalLayer.size(); i++){
			//in Duda, this update is: delta = f'(net) * (t_k - z_k)
			finalLayer[i].Delta = finalLayer[i].PhiPrime() * (example.back() - finalLayer[i].Output);
		}
		
		//backpropagate the deltas through the layers
		for(l = _layers.size() - 2; l >= 0; l--){
			vector<Neuron>& leftLayer = _layers[l];
			vector<Neuron>& rightLayer = _layers[l+1];
			for(i = 0; i < leftLayer.size(); i++){
				//sum products over the deltas/weights from the right layer
				for(j = 0, sum = 0.0; j < rightLayer.size(); j++){
					sum += (rightLayer[j].Delta * rightLayer[j].Weights[i+1]); //plus one on the right, to account for the zeroeth bias weight
				}
				leftLayer[i].Delta = leftLayer[i].PhiPrime() * sum;
			}
		}

		/*
		//update weights based on deltas just calculated; this is the actual gradient update part
		//update the final layer weights first
		vector<Neuron>& prevNeurons = _layers[_layers.size()-2];
		vector<Neuron>& outputNeurons = _layers[_layers.size()-1];
		for(i = 0; i < outputNeurons.size(); i++){ //iterate the output neurons; the number of weights on these neurons is equal to the number of neurons in the previous layer + 1
			//iterate the weights for this output neuron; the first/0th weight is always the bias
			for(j = 0; j < outputNeurons[i].Weights.size(); j++){
				if(j == 0){ //update the bias weight, which is attached to no previous neuron; the 'input' is implicitly a constant 1.0
					dW = (_eta * outputNeurons[i].Delta);
				}
				else{ //else, for all other weights the update is dependent on the output of a neuron in previous layer
					dW = (_eta * outputNeurons[i].Delta * prevNeurons[j-1].Output);
				}
				outputNeurons[i].Weights[j] += dW;
			}
		}
		*/
		
		//cout << "done4" << endl;
		//now iterate and update the weights, from output layer to first hidden layer
		for(l = _layers.size()-1; l >= 0; l--){
			//iterate the neurons in this layer
			for(i = 0; i < _layers[l].size(); i++){
				//iterate the weights for this neuron
				for(j = 0; j < _layers[l][i].Weights.size(); j++){
					if(j == 0){ //for j==0, the bias weight is not associated with a previous neuron; the input is implicitly a constant 1.0
						dW = _eta * _layers[l][i].Weights[0] * _layers[l][i].Delta;
					}
					else{
						//if this is the first hidden layer, the input comes from the example, rather than a previous neuron layer
						if(l == 0){
							input = example[j-1]; //minus one to account for the bias
						}
						//else, it comes from the output of the previous layer's (j-1)th neuron
						else{
							input = _layers[l-1][j-1].Output;  //minus one, for the sake of skipping the bias weight
						}
						dW = _eta * _layers[l][i].Delta * input;
					}
					//weight update
					//printf("%f\n",_layers[l][i].Weights[j]);
					//cout << "layer " << l << " update " << i << "," << j << "  : " << dW << endl;
					_layers[l][i].Weights[j] += dW;
					//printf("%f\n",_layers[l][i].Weights[j]);
				}
			}
		}
		//cout << "done5" << endl;
		
		
		//track error info
		prevNetError = netError;
		netError = 0.0;
		for(i = 0; i < _layers[_layers.size()-1].size(); i++){
			netError += _layers[_layers.size()-1][i].Delta;
		}
		//save this error level
		errorHistory[ringIndex] = netError;
		ringIndex = (ringIndex + 1) % 50;
		
		iterations++;
		cout << "\r" <<  iterations << ") Error: " << netError << "\tDelta: " << (prevNetError - netError) << "                  " << flush;
		//cout << "Example: " << example.back() << "  Output: " << _layers[_layers.size()-1][0].Output << endl;
		//iterations++;
		//if(iterations % 50 == 49){
		//	//average the error of the last 50 examples learned
		//	double avgError = 0.0;
		//	for(i = 0; i < errorHistory.size(); i++){
		//		avgError += errorHistory[i];
		//	}
		//	avgError /= 50.0;
		//	cout << "Iteration " << iterations << " avg error: " << avgError << endl;
		//	///cin >> dummy;
		//}
		
		//update _eta, the learning rate
	}
}

/*
Network outputs need not be binary, so the caller must read Output vector to get the classification result.
All this function does is set the input pointers to the example values, then calls Stimulate() layer by layer
in feed forward fashion to propagate the input.

@example: a vector feature values of at least d-dimensions, where d is the number of inputs to the network.
*/
void MultiLayerNetwork::Classify(const vector<double>& example)
{
	int l, i, j;
	
	/* assume this was done only once, in the network configuration
	//rig the first bias, in case it wasn't elsewhere
	for(i = 0; i < _layers[0].size(); i++){
		_layers[0][i].Inputs[0] = &_biases[0];
	}
	*/
	
	//fix the first layer's inputs to the example
	for(i = 0; i < _layers[0].size(); i++){
		for(j = 1; j < _layers[0][i].Inputs.size(); j++){
			_layers[0][i].Inputs[j] = &example[j-1]; //minus one, to offset for the bias
		}
	}
	
	//feed-forward: stimulate the neurons, layer by layer left-to-right
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].Stimulate();
		}
	}
	
	//don't return anything, just let the caller read the outputs, for now.
}

//Client reads the network output by simply reading the output neurons for themselves.
const vector<Neuron>& MultiLayerNetwork::GetOutput()
{
	return _layers.back();
}

/*
Tests a dataset, and outputs the predicting for each vector to the file given by the outputPath.

@testSet: A set of d-dimensional vectors with no class labels
@outputPath: The output path to which each testSet vector shall be written in csv, appended with its
predicted class label. 
*/
void MultiLayerNetwork::Test(const string& outputPath, vector<vector<double> >& testSet)
{
	string temp;
	char buf[64] = {'\0'};
	ofstream outputFile(outputPath, ios::out);

	for(int i = 0; i < testSet.size(); i++){
		Classify(testSet[i]);
		
		temp.clear();
		for(int j = 0; j < testSet[i].size(); j++){
			sprintf(buf,"%lf,",testSet[i][j]);
			temp += buf;
		}
		
		sprintf(buf,"%lf\n",_layers.back()[0].Output);
		temp += buf;
		
		outputFile << temp;
	}
	
	outputFile.close();
}

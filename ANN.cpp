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

//a builder; this could be moved to some builder class
void MultiLayerNetwork::_buildTwoLayerNet(int numInputs, int numHiddenUnits, int numOutputUnits)
{
	int i, j;
	
	//clear any existing model
	Clear();
	
	//just two layers for now
	_layers.resize(2);
	
	//set up the hidden units
	_layers[0].resize(numHiddenUnits, Neuron(numInputs, TANH) );
	_nullifyLayer(_layers[0]);
	
	//set up the output units, rigging their inputs to the outputs of the previous hidden layer
	_layers[_layers.size()-1].resize(numOutputUnits, Neuron(numHiddenUnits, TANH) );
	vector<Neuron>& finalLayer = _layers[_layers.size() - 1];
	vector<Neuron>& previousLayer = _layers[_layers.size() - 2];
	for(i = 0; i < finalLayer.size(); i++){
		for(j = 0; j < previousLayer.size(); j++){
			//assign the (j+1)th previous neuron's output to the ith neuron's jth input; the +1 is to account for the bias
			finalLayer[i].Inputs[j+1] = &previousLayer[j].Output;
		}
	}

	//set up the bias weights; the zeroeth input of every neuron will be a bias weight
	_biases.resize(2);
	//set the first layer bias ptrs
	for(i = 0; i < _layers[0].size(); i++){
		_layers[0][i].Inputs[0] = &_biases[0];
	}
	//set the final layer bias ptrs
	for(i = 0; i < finalLayer.size(); i++){
		finalLayer[i].Inputs[0] = &_biases[_biases.size()-1];
	}
}

/*
There are specific strategies for initializing the weights (see Haykin). Here they are just init'ed with random numbers.
*/
void MultiLayerNetwork::_assignRandomWeights()
{
	int i, j, l;
	
	srand(time(NULL));
	
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
		_biases[i] = ((double)(rand() % 100)) / 50.0;
		if(rand() % 2 == 0){
			_biases[i] *= -1; //flip the sign 50% of the time
		}
	}
}



/*
Runs the backpropagation algorithm from Duda. Also see Mustafa's Neural Nets CalTech lecture,
which this mirrors.

Another very terse source: http://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html

The gist:
	Iterate over the examples e, doing the following:
	Forward compute the activation for all neurons using the e
	-Calculate the error at the output nodes: delta_o = (t_k - z_k) * phi_prime(signal)
	-Calculate the error at the hidden layer: dW = phi_prime(signal) * Sigma( delta_o * w_io )
	-Update the weights for the hidden layer, using these dW's
	-Update the weights for the final layer, using delta
	-Iterate, until the error for the network falls below some threshold. The most straightforward error is simply the sum-of-squares of the delta_o
*/
void MultiLayerNetwork::BackPropagation(const vector<vector<double> >& dataset, double eta)
{
	int i, j, l;
	double convergence, input, netError, prevNetError, sum, dW;

	
	//intialize the weights to random values
	_assignRandomWeights();

	netError = 1;
	convergence = 0;
	//while(netError > convergenceThreshold){
	while(true){
		//randomly choose an example
		const vector<double>& example = dataset[ rand() % dataset.size() ];
		//Run the forward pass of the network to set all neuron outputs for this sample
		Classify(example);

		//calculate the final-layer deltas, which are just the signal-prime values
		vector<Neuron>& finalLayer = _layers[_layers.size()-1];
		for(i = 0; i < finalLayer.size(); i++){
			//in Duda, this update is: delta = f'(net) * (t_k - z_k)
			finalLayer[i].Delta = finalLayer[i].TanhPrime( finalLayer[i].Signal ) * (example.back() - finalLayer[i].Output);
		}
		
		//backpropagate the deltas through the layers
		for(l = _layers.size() - 2; l >= 0; l--){
			vector<Neuron>& leftLayer = _layers[l];
			vector<Neuron>& rightLayer = _layers[l+1];
			for(i = 0; i < leftLayer.size(); i++){
				//sum products over the deltas/weights from the right layer
				for(j = 0, sum = 0.0; j < rightLayer.size(); j++){
					sum += (rightLayer[j].Delta * rightLayer[j].Weights[i+1]); //plus one on the right, to account for the bias weight
				}
				leftLayer[i].Delta = leftLayer[i].TanhPrime( leftLayer[i].Signal ) * sum;
			}
		}

		//update weights based on deltas just calculated; this is the actual gradient update part
		//update the final layer weights first
		vector<Neuron>& prevNeurons = _layers[_layers.size()-2];
		vector<Neuron>& outputNeurons = _layers[_layers.size()-1];
		for(i = 0; i < outputNeurons.size(); i++){ //iterate the output neurons; the number of weights on these neurons is equal to the number of neurons in the previous layer + 1
			//iterate the weights for this output neuron; the first/0th weight is always the bias
			for(j = 0; j < outputNeurons[i].Weights.size(); j++){
				if(j == 0){ //update the bias weight, which is attached to no previous neuron
					dW = (eta * outputNeurons[i].Delta);
				}
				else{ //else, for all other weights the update is dependent on the output of a neuron
					dW = (eta * outputNeurons[i].Delta * prevNeurons[j-1].Output);
				}
				outputNeurons[i].Weights[j] += dW;
			}
		}
		
		//now iterate and update the hidden layer weights
		for(l = _layers.size()-2; l <= 0; l--){ //iterate the neuron layers
			vector<Neuron> hiddenLayer = _layers[l];
			//iterate the neurons in this layer
			for(i = 0; i < hiddenLayer.size(); i++){
				//iterate the weights for this neuron
				for(j = 0; j < hiddenLayer[i].Weights.size(); j++){
					if(j == 0){ //for j==0, the bias weight is not associated with a previous neuron; the input is implicitly a constant 1.0
						dW = eta * hiddenLayer[i].Weights[j] * hiddenLayer[i].Delta;
					}
					else{
						//if this is the first hidden layer, the input comes from the example; else, it comes from the previous layer's output
						if(l == 0){
							input = example[j-1];  //minus one, for the sake of skipping the bias weight
						}
						else{
							input = hiddenLayer[j-1].Signal;  //minus one, for the sake of skipping the bias weight
						}
						dW = eta * hiddenLayer[i].Weights[j] * hiddenLayer[i].Delta * input;
					}
					//weight update
					hiddenLayer[i].Weights[j] += dW;
				}
			}
		}

		//track error info
		prevNetError = netError;
		netError = 0.0;
		for(i = 0; i < _layers[_layers.size()-1].size(); i++){
			netError += _layers[_layers.size()-1][i].Delta;
		}
		cout << "Error: " << netError << "\tDelta: " << (prevNetError - netError) << endl;
		
		//update eta, the learning rate
		
	}
}

/*
Network outputs need not be binary, so the caller must read Output vector to get the classification result.

@example: a vector of feature values, along with a +/-1 to indicate class membership.
*/
void MultiLayerNetwork::Classify(const vector<double>& example)
{
	int l, i, j;
	
	//fix the first layer's inputs to the example
	for(i = 0; i < _layers[0].size(); i++){
		for(j = 0; j < _layers[0][i].Inputs.size(); j++){
			_layers[0][i].Inputs[j+1] = &example[j]; //plus one, to offset for the bias
		}
	}
	
	//stimulate the neurons, layer by layer left-to-right
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].CalculateOutput();
		}
	}
	
	//don't return anything, just let the caller read the outputs, for now.
}

void MultiLayerNetwork::Test(vector<vector<double> >& dataset)
{
	
	for(int i = 0; i < dataset.size(); i++){
		Classify(dataset[i]);
	}
}

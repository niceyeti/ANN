#include "ANN.hpp"

MultiLayerNet::MultiLayerNet(int numInputs, int numHiddenUnits, int numOutputUnits)
{
	//set up the hidden units
	for(int i = 0; i < numHiddenUnits; i++){
		_hiddenUnits.push_back( Neuron(numInputs, TANH) );
	}

	//set up the hidden units
	for(int i = 0; i < numHiddenUnits; i++){
		_outputUnits.push_back( Neuron(numHiddenUnits, TANH) );
	}	
}

MultiLayerNet::~MultiLayerNet()
{
	_hiddenUnits.clear();
	_outputUnits.clear();
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
void BackPropagation(vector<vector<double> >& dataset, double eta)
{
	int i, j, l;
	double convergenceThreshold, netError, prevNetError, sum, dW;

	//intialize the weights to random values
	_assignRandomWeights();

	networkError = 1;
	convergence = 0;
	while(networkError > convergenceThreshold){
		//randomly choose an example
		vector<double>& example = dataset[ rand() % dataset.size() ];
		//Run the forward pass of the network to set all neuron outputs for this sample
		Classify(example);

		//calculate the final-layer deltas, which are just the signal-prime values
		vector<Neuron>& finalLayer = _layers[_layers.size()-1];
		for(i = 0; i < finalLayer.size(); i++){
			//in Duda, this update is: delta = f'(net) * (t_k - z_k)
			finalLayer[i].Delta = finalLayer[i].TanhPrime( finalLayer[i].Signal ) * (example.last() - finalLayer[i].Output);
		}
		
		//backpropagate the deltas through the layers
		for(l = _layers().size() - 2; l >= 0; l--){
			vector<Neuron> leftLayer = _layers[l];
			vector<Neuron> rightLayer = _layers[l+1];
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
			for(i = 0; i < lefttLayer.size(); i++){
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
		for(i = 0; i < _layers[_layer.size()]; i++){
			netError += _layers[i].Delta;
		}
		cout << "Error: " << netError << "\tDelta: " << (prevNetError - netError) << endl;
		
		//update eta, the learning rate
		
		
	}
}

/*
Network outputs need not be binary, so the caller must read Output vector to get the classification result.

@example: a vector of feature values, along with a +/-1 to indicate class membership.
*/
void Classify(vector<double>& example)
{
	int l, i, j;
	
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[0].size(); i ++){
			_layers[l][i].Simulate(example);
		}
	}
	
	//don't return anything, just let the caller read the outputs, for now.
	
}

void Test(vector<vector<double> >& dataset)
{
	int output;
	
	for(int i = 0; i < dataset.size(); i++){
		Classify(dataset[i]);
	}
}


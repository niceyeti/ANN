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

Imlements stochastic gradient descent, using standard backpropagation.
*/
void BackPropagation(vector<vector<double> >& dataset, double eta)
{
	double convergenceThreshold, networkError;


	
	//intialize the weights to random values
	_assignRandomWeights();
	
	networkError = 1;
	convergence = 0;
	while(networkError > convergenceThreshold){
		//randomly choose an example
		vector<double>& example = dataset[ rand() % dataset.size() ];
		//Run the forward pass of the network to set all neuron outputs for this sample
		Classify(example);
		//backpropagate the error 
		//BackPropagate();
		
		//calculate error at the hidden layer
		for(l = _layers.size() - 1; l >= 0; l--){

			
			//calculate the final-layer deltas, which are just the signal-prime values
			vector<Neuron>& rightLayer = _layers[_layers.size()-1];
			for(i = 0; i < rightLayer.size(); i++){
				rightLayer[i].Delta = rightLayer[i].TanhPrime( rightLayer[i]. );	
			}
			//backpropagate the deltas through the layers
			for(l = _layers().size() - 2; l >= 0; l--){
				vector
				
				//sum products over the deltas/weights from the right layer
				for(j = 0, sum = 0.0; j < rightLayer.size(); j++){
					sum += (rightLayer[].Delta * _weights[l][j]); 
				}
				leftLayer[i].Delta = leftLayer[i].TanhPrime( leftLayer[i].Signal ) * sum;
			}
			
			
			
		}
		
		
		
		
		
		for(l = 0; l < layers.size(); l++){
			
		}
		
		
	}
}

/*
Network outputs need not be binary, so the caller must read Output vector to get the classification result.

@example: a vector of fature values, along with a +/-1 to indicate class membership.
*/
void Classify(vector<double>& example)
{
	//set the network inputs
	for(int i = 0; i < _inputs.size(); i++){
		_inputs[i] = example[i];
	}
	
	//calculate the first hidden layer signals and outputs
	for(int i = 0; i < _hiddenUnits.size(); i++){
		_hiddenUnits[i].
		
		
	}
	
	
	
	
	HiddenOutputs.resize(_hiddenUnits.size());
	Outputs.resize();
	for(int i = 0; i < _hiddenUnits.size(); i++){
		
		
		HiddenOutputs[i] = _hiddenUnits[i].Simulate(example);
	}
	
	for(int i = 0; i < _outputUnits.size(); i++){
		Outputs[i] = _outputUnits[i].Simulate(hiddenOutputs);
	}
}

void Test(vector<vector<double> >& dataset)
{
	int output;
	
	for(int i = 0; i < dataset.size(); i++){
		Classify(dataset[i]);
	}
}





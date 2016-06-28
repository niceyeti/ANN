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

Imlements stochastic gradient descent, using standard backpropagation.


*/
void BackPropagation(vector<vector<double> >& dataset, double eta)
{
	int i, j, l;
	double convergenceThreshold, networkError, sum;

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
		//_backPropagate();
		//calculate the final-layer deltas, which are just the signal-prime values
		vector<Neuron>& finalLayer = _layers[_layers.size()-1];
		for(i = 0; i < finalLayer.size(); i++){
			//in Duda, this update is: delta = f'(net) * (t_k - z_k)
			finalLayer[i].Delta = finalLayer[i].TanhPrime( finalLayer[i].Signal ) * (example.last() - finalLayer[i].Output);
		}
		
		//backpropagate the deltas through the layers (here l, i, and j correspond with the notation by Abu-Mustafa)
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
					outputNeurons[i].Weights[j] = outputNeurons[i].Weights[j] + eta * outputNeurons[i].Delta;
				}
				else{ //else, for all other weights the update is dependent on the output of a neuron
					outputNeurons[i].Weights[j] = outputNeurons[i].Weights[j] + eta * outputNeurons[i].Delta * prevNeurons[j-1].Output;
				}
			}
		}
		
		/*
		//TODO: Code for iterating multiple hidden layers; not needed for now, for a two-layer net of 1 hidden layer, 1 output layer
		//now iterate the hidden layers, recursively
		for(l = _layers.size()-2; l <= 0; l--){ //iterate the neuron layers
			vector<Neuron> leftLayer = _layers[l];
			vector<Neuron> rightLayer = _layers[l+1];
			for(i = 0; i < rightLayer.size(); i++){ //iterate the neurons in this layer
				Neuron& rightNeuron = rightLayer[i];
				for(j = 0; j < rightNeuron.Weights.size(); j++){
					if(j == 0){ //again, for j==0, the bias weight is not associated with a previous neuron; the input is implicitly a constant 1.0
						rightNeuron.Weights[j] = rightNeuron.Weights[j] + eta * rightNeuron.Weights[j] * rightNeuron.Delta;
					}
					else{
						rightNeuron.Weights[j] = rightNeuron.Weights[j] + eta * rightNeuron.Weights[j] * rightNeuron.Delta * leftLayer[j-1].Delta; //minus one, for the sake of skipping the bias weight
					}
				}
			}
		}
		*/
		
		//Iterate the first (leftmost) hidden layer; this code should remain active even if there are multiple hidden layers
		vector<Neuron>& firstLayer = _layers[0];
		for(i = 0; i < firstLayer.size(); i++){ //iterate the neurons in this layer
			for(j = 0; j < firstLayer[i].Weights.size(); j++){
				if(j == 0){ //update the the bias weight
					firstLayer[i].Weights[j] = firstLayer[i].Weights[j] + eta * firstLayer[i].Delta;
				}
				else{
					firstLayer[i].Weights[j] = firstLayer[i].Weights[j] + eta * firstLayer[i].Delta * example[j-1];
				}
			}
		}
		
		//update eta
		
		//caculate total change in weights to detect convergence threshold
		
		
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





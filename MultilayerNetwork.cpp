#include "MultilayerNetwork.hpp"

MultilayerNetwork::MultilayerNetwork()
{
	_momentum = 0;
	_eta = 0;
}

MultilayerNetwork::MultilayerNetwork(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits)
{
	_momentum = 0;
	_eta = 0.1;

	BuildNet(numInputs, numLayers, numHiddenUnits, numOutputUnits);
}

MultilayerNetwork::~MultilayerNetwork()
{
	_layers.clear();
}

void MultilayerNetwork::SetHiddenLayerFunction(ActivationFunction activationType, int layer)
{
	for(int i = 0; i < _layers[layer].size(); i++){
		_layers[layer][i].PhiFunction = activationType;
	}
}

void MultilayerNetwork::SetOutputLayerFunction(ActivationFunction activationType)
{
	for(int i = 0; i < _layers[_layers.size()-1].size(); i++){
		_layers[_layers.size()-1][i].PhiFunction = activationType;
	}
}

void MultilayerNetwork::Clear()
{
	_layers.clear();
	_biases.clear();
}

void MultilayerNetwork::SetEta(double newEta)
{
	_eta = newEta;
}

//Just a utility for nullifying the input ptrs of some layer
void MultilayerNetwork::_nullifyLayer(vector<Neuron>& layer)
{	
	for(int i = 0; i < layer.size(); i++){
		for(int j = 0; j < layer[i].Inputs.size(); j++){
			layer[i].Inputs[j] = NULL;
		}
	}
	
}

/*
A builder; this could be moved to some builder class

@numInputs: The dimension of the input data, such as 2 or 3-d. The network neurons will be configured with numInputs + 1 inputs,
where the additional input is for the bias.
@numInputs: number of inputs (attributes)
@numLayers: Number of layers. This should always be two, but its fun to experiment...
@numHiddenUnits: Number of hidden units in the hidden layer
@numOutputUnits: Nmber of output units
*/
void MultilayerNetwork::BuildNet(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits)
{
	int i, j, l;

	//clear any existing model
	Clear();

	//init the layers
	_layers.resize(numLayers);

	//lay out the hidden units
	for(i = 0; i < (numLayers-1); i++){
		_layers[i].resize(numHiddenUnits, Neuron(numInputs + 1, TANH) );  //plus one for the biases
		_nullifyLayer(_layers[i]);
	}

	//set up the output units
	_layers[_layers.size()-1].resize(numOutputUnits, Neuron(numHiddenUnits + 1, TANH) ); //plus one for the biases
	_nullifyLayer(_layers[1]);

	//connect the layers
	for(l = numLayers - 1; l > 0; l--){
		vector<Neuron>& prevLayer = _layers[l-1];
		vector<Neuron>& curLayer = _layers[l];
		for(i = 0; i < curLayer.size(); i++){
			for(j = 0; j < prevLayer.size(); j++){
				//assign the (j+1)th previous neuron's output to the ith neuron's jth input; the +1 is to account for the bias
				curLayer[i].Inputs[j+1] = &prevLayer[j].Output;
			}
		}
	}

	//connect the bias inputs; the zeroeth input of every neuron will be a bias weight
	_biases.resize(numLayers);
	for(l = numLayers-1; l >= 0; l--){
		vector<Neuron>& layer = _layers[l];
		for(i = 0; i < layer.size(); i++){
			layer[i].Inputs[0] = &_biases[l];
		}
	}

	//initialize the biases to constant 1.0
	for(i = 0; i < _biases.size(); i++){
		_biases[i] = 1.0;
	}
}

//There are many neural init strategies; this is the dumbest.
void MultilayerNetwork::InitializeWeights()
{
	//intialize the weights to random values
	_assignRandomWeights();
}

/*
Prints weights of the entire network.
*/
void MultilayerNetwork::PrintWeights()
{
	int i, j, l;
	
	//print each layer
	for(l =0; l < _layers.size(); l++){
		//print each neuron in this layer
		cout << "Layer " << l << " weights: " << endl;
		for(i = 0; i < _layers[l].size(); i++){
			//print each weight for this neuron
			cout << "  neuron " << i << ": " << flush;
			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				//cout << _layers[l][i].Weights[j].w << " ";
				printf("%03.3f ",_layers[l][i].Weights[j].w);
			}
			cout << endl;
		}
	}
}


/*
There are specific strategies for initializing the weights (see Haykin). Here they are just init'ed with random numbers.
*/
void MultilayerNetwork::_assignRandomWeights()
{
	int i, j, l;
	
	//TODO: change the assignment to a zero-mean Gaussian, or other init per lit recommendations.
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				_layers[l][i].Weights[j].w = ((double)(rand() % 100)) / 50.0;
				if(rand() % 2 == 0){
					_layers[l][i].Weights[j].w *= -1.0; //flip the sign 50% of the time
				}
			}
		}
	}
	
	//init the biases to all 1.0
	for(i = 0; i < _biases.size(); i++){
		_biases[i] = 1.0;
	}
}

/*
Momentum must lie between 0 and 1.0 for convergence.
Recommend value is ~0.5. See Haykin.

Momentum doesn't just speed up updates, it is claimed to help overcome local minima, as opposed
to using no momentum.
*/
void MultilayerNetwork::SetMomentum(double momentum)
{
	if(momentum > 1.0){
		cout << "WARNING momentum values > 1.0 will cause weights to diverge" << endl;
	}

	_momentum = momentum;
}

//Validates output of network is not nan, inf, etc. This is critical so these
//values aren't backpropagated, which destroys the learned weights.
bool MultilayerNetwork::IsOutputNormal()
{
	vector<Neuron>& finalLayer = _layers[_layers.size()-1];
	for(int i = 0; i < finalLayer.size(); i++){
		if(!std::isnormal(finalLayer[i].Output)){
			return false;
		}
	}

	return true;
}

/*
Given that Classify() has been called for an example, this backpropagates the error given
by that single example. Hence this function is stateful, in that it assumes the network
outputs have been driven by a specific example. The reason this is public is for clients
that do online learning, like in approximate Q-learning.
*/
void MultilayerNetwork::BackpropagateError(const vector<double>& inputs, double target)
{
	int i, j, l;
	double sum;

	//prevent nans, inf, etc from being backpropagated
	if(!std::isnormal(target) || !IsOutputNormal()){
		cout << "ERROR target ("<< target << ") or one or more outputs abnormal, ABORTING BACKPROP" << endl;
		return;
	}

	//calculate the final-layer deltas, which are just the signal-prime values
	vector<Neuron>& finalLayer = _layers[_layers.size()-1];
	for(i = 0; i < finalLayer.size(); i++){
		//in Duda, this update is: delta = f'(net) * (t_k - z_k)
		finalLayer[i].Delta = finalLayer[i].PhiPrime() * (target - finalLayer[i].Output);
	}

	//backpropagate the deltas through the layers
	for(l = _layers.size() - 2; l >= 0; l--){
		vector<Neuron>& leftLayer = _layers[l];
		vector<Neuron>& rightLayer = _layers[l+1];
		for(i = 0; i < leftLayer.size(); i++){
			//sum products over the deltas/weights from the right layer
			for(j = 0, sum = 0.0; j < rightLayer.size(); j++){
				sum += (rightLayer[j].Delta * rightLayer[j].Weights[i+1].w); //plus one on the right, to account for the zeroeth bias weight
			}
			leftLayer[i].Delta = leftLayer[i].PhiPrime() * sum;
		}
	}
}

/*
Like Backpropagate(), this publicly exposes the weight-update step, after the network outputs have been
driven with Classify() and the error back-propagated by Backpropagate(). Again, exposing this publicly
is meant as a convenience for online learners, like in Q-learning.
*/
void MultilayerNetwork::UpdateWeights(const vector<double>& inputs, double target)
{
	double dw, input;

	if(!IsOutputNormal()){
		cout << "ERROR one or more outputs abnormal, ABORTING UPDATEWEIGHTS()" << endl;
		return;
	}

	//now iterate and update the weights, from output layer to first hidden layer
	for(int l = _layers.size()-1; l >= 0; l--){
		//iterate the neurons in this layer
		for(int i = 0; i < _layers[l].size(); i++){
			//iterate the weights for this neuron
			for(int j = 0; j < _layers[l][i].Weights.size(); j++){
				if(j == 0){ //for j==0, the bias weight is not associated with a previous neuron; the input is implicitly a constant 1.0
					dw = _eta * _layers[l][i].Weights[0].w * _layers[l][i].Delta;
				}
				else{
					//if this is the first hidden layer, the input comes from the input, rather than a previous neuron layer
					if(l == 0){
						input = inputs[j-1]; //minus one to account for the bias
					}
					//else, it comes from the output of the previous layer's (j-1)th neuron
					else{
						input = _layers[l-1][j-1].Output;  //minus one, for the sake of skipping the bias weight
					}
					dw = _eta * _layers[l][i].Delta * input;
				}
				//weight update
				//printf("%f\n",_layers[l][i].Weights[j]);
				//cout << "layer " << l << " update " << i << "," << j << "  : " << dW << endl;
				if(_momentum == 0){ //no momentum, so ignore previous dw
					_layers[l][i].Weights[j].w += dw;
				}
				else{
					//see Haykin, Neural Nets. For momentum, the previous dw is used to add momentum to the weight update.
					_layers[l][i].Weights[j].w = _momentum * _layers[l][i].Weights[j].dw + dw + _layers[l][i].Weights[j].w;
					_layers[l][i].Weights[j].dw = dw;
				}
				//printf("%f\n",_layers[l][i].Weights[j]);
			}
		}
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
void MultilayerNetwork::BatchTrain(const vector<vector<double> >& dataset)
{
	string dummy;
	int iterations, minIteration, ringIndex;
	double convergence, input, netError, minError, prevNetError;
	vector<double> errorHistory;
	
	//initialize the weights to random values
	InitializeWeights();
	
	//sliding error-window for viewing long term error changes
	errorHistory.resize(50, 0.0);
	ringIndex = 0;
	
	iterations = 0;
	netError = 1;
	minError = 10000000;
	convergence = 0;
	_eta = 0.1;
	_momentum = 0.0; //reportedly a good value (see Haykin)

	//while(netError > convergenceThreshold){
	while(iterations < 500000){
		//PrintWeights();
		
		//randomly choose an example
		const vector<double>& example = dataset[ rand() % dataset.size() ];
		//learns from a single example: drives the network outputs, backprops, and updates the weights
		Backpropagate(example);
		
		//track error info
		prevNetError = netError;
		netError = GetNetError();
		//save this error level
		errorHistory[ringIndex] = abs(netError);
		ringIndex = (ringIndex + 1) % (int)errorHistory.size();
		
		iterations++;
		if(iterations % 50 == 49){
			//average the error of the last 50 examples learned
			double avgError = 0.0;
			for(int i = 0; i < errorHistory.size(); i++){
				avgError += errorHistory[i];
				errorHistory[i] = 0;
			}
			avgError /= (double)errorHistory.size();
			//cout << "\rIteration " << iterations << " avg error: " << avgError << "                " << flush;
			cout << "Iteration " << iterations << " avg error: " << avgError << "                " << endl;
			if(avgError < minError){
				minError = avgError;
				minIteration = iterations;
			}
		}
		
		//update _eta, the learning rate
	}

	cout << "Minimum error: " << minError << " at iteration " << minIteration << endl;
}

/*
Just a convenient wrapper for learning from a single example: drive the outputs, backpropagate
the error, and update the weights based off the single example.
*/
void MultilayerNetwork::Backpropagate(const vector<double>& example)
{
	if(IsValidExample(example)){
		//Run the forward pass of the network to set all neuron outputs for this sample
		Classify(example);
		//back propagate the error measure/gradient
		BackpropagateError(example, example.back());
		//Update the weights based on the backpropagated error values
		UpdateWeights(example, example.back());
	}
}

bool MultilayerNetwork::IsValidExample(const vector<double>& example)
{
	for(int i = 0; i < example.size(); i++){
		if(!std::isnormal(example[i])){
			cout << "WARNING In MultilayerNetwork, isnormal() returned for input example value example[" << i << "] of " << example.size() << " length vector" << endl;
			cout << example[i] << endl;
			cout << FpClassify(example[i]) << endl;
			return false;
		}
	}
	return true;
}

/*
Assuming Backpropagate() has been called after the network has been driven by some example, this
just sums across the error across all output nodes.

In the future this could be factored or paramterized somehow to support error measures for
different gradient definitions.
*/
double MultilayerNetwork::GetNetError()
{
	double netError = 0.0;

	for(int i = 0; i < _layers[_layers.size()-1].size(); i++){
		netError += _layers[_layers.size()-1][i].Delta;
	}

	return netError;
}

/*
Network outputs need not be binary, so the caller must read Output vector to get the classification result.
All this function does is set the input pointers to the example values, then calls Stimulate() layer by layer
in feed forward fashion to propagate the input.

@inputs: a vector feature values of at least d-dimensions, where d is the number of inputs to the network. The
number of network input nodes determines how much will be read from @input, so for instance its okay to pass an
example vector longer than the number of inputs where the last item in the example is the teacher's target value.
*/
void MultilayerNetwork::Classify(const vector<double>& inputs)
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
			_layers[0][i].Inputs[j] = &inputs[j-1]; //minus one, to offset for the bias
		}
	}
	
	//feed-forward: stimulate the neurons, layer by layer left-to-right
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].Stimulate();
		}
	}
	
	//don't return anything, just let the caller read the outputs, for now.

	ValidateOutputs();
}

//Checks network outputs for NAN, INF, etc
void MultilayerNetwork::ValidateOutputs()
{
	//dbg: warn of NAN and other erroneous float values
	for(int i = 0; i < _layers.back().size(); i++){
		if(!std::isnormal(_layers.back()[i].Output)){
			cout << "WARNING isnormal() returned false for output neuron " << i << endl;
			cout << FpClassify(_layers.back()[i].Output) << endl;
		}
	}
}

//Prints the fp-error condition
const char* MultilayerNetwork::FpClassify(double x)
{
    switch(std::fpclassify(x)){
        case FP_INFINITE:  return "Inf";
        case FP_NAN:       return "NaN";
        case FP_NORMAL:    return "normal";
        case FP_SUBNORMAL: return "subnormal";
        case FP_ZERO:      return "zero";
        default:           return "unknown";
    }
}

//Client reads the network output by simply reading the output neurons for themselves.
const vector<Neuron>& MultilayerNetwork::GetOutputs()
{
	return _layers.back();
}

/*
Tests a dataset, and outputs the predicting for each vector to the file given by the outputPath.

@testSet: A set of d-dimensional vectors with no class labels
@outputPath: The output path to which each testSet vector shall be written in csv, appended with its
predicted class label.
*/
void MultilayerNetwork::Test(const string& outputPath, vector<vector<double> >& testSet)
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

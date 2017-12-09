#include "MultilayerNetwork.hpp"

TrainingExample::TrainingExample(const vector<double>& state, double rewardTarget)
{
	xs = state;
	target = rewardTarget;
}

MultilayerNetwork::MultilayerNetwork()
{
	srand(time(NULL));
	
	_momentum = 0;
	_eta = 0;
	_weightDecayRate = 0;
	_useRegularizer = false;
}

MultilayerNetwork::MultilayerNetwork(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits)
{
	srand(time(NULL));

	_momentum = 0;
	_eta = 0.1;
	_weightDecayRate = 0;

	BuildNet(numLayers, numInputs, numHiddenUnits, numOutputUnits);
}

MultilayerNetwork::~MultilayerNetwork()
{
	_layers.clear();
}

/*
A simple L2 regularization method in nn's is to reduce the weight update value at each step.

The update equation (ignoring all of the math) becomes:
	w = w + d_w - lambda * w
*/
void MultilayerNetwork::SetWeightDecay(double decayRate)
{
	if(decayRate < 0 || decayRate > 1.0){
		cout << "ERROR setting weight decay parameter to out of range value, chaos ensues: " << decayRate;
	}

	_weightDecayRate = decayRate;
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
Builds a 'deep' network of fully-connected layers using the schema provided by the function parameters. The only difference
is the higher level of parameterization; this could eliminate BuildNet().

@neuronsPerLayer: The number of neurons per layer, indexed via 0=first hidden layer, 1=second hidden layer, etc.
@initialBias: The initial value for the biases. If 0.0, then the model will have effectively no bias.
*/
void MultilayerNetwork::BuildDeepNetwork(int numInputs, int numLayers, vector<int> neuronsPerLayer, vector<ActivationFunction> activationSchema, double initialBias)
{
	int i, j, l, numLayerInputs;

	//clear any existing model
	Clear();

	//init the neuron layers
	_layers.resize(numLayers);

	//lay out the layers
	for(i = 0; i < numLayers; i++){
		//set number of inputs to either the number of network inputs, or the number of neurons in the previous layer (for successive hidden layers)
		if(i == 0){
			numLayerInputs = numInputs;
		}
		else{
			numLayerInputs = _layers[i-1].size();
		}
		
		_layers[i].resize(neuronsPerLayer[i], Neuron(numLayerInputs + 1, activationSchema[i]) );  //plus one for this layer's bias
		_nullifyLayer(_layers[i]); //nullify this layer's input pointers before connection in next steps
	}

	//set up the inter-layer connections, fully connected
	for(l = numLayers - 1; l > 0; l--){
		vector<Neuron>& prevLayer = _layers[l-1];
		vector<Neuron>& curLayer = _layers[l];
		for(i = 0; i < curLayer.size(); i++){
			for(j = 0; j < prevLayer.size(); j++){
				//assign the (j+1)th previous neuron's output to the ith neuron's jth input; the +1 is to account for the bias in the zeroeth index
				curLayer[i].Inputs[j+1] = &prevLayer[j].Output;
			}
		}
	}

	//connect the bias inputs; the zeroeth input of every neuron will be a bias weight
	_biases.resize(numLayers);
	for(l = _layers.size()-1; l >= 0; l--){
		//init the 0th input of every neuron in this layer to the bias for the layer
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].Inputs[0] = &_biases[l];
		}
	}

	//initialize the biases to constant 1.0
	for(i = 0; i < _biases.size(); i++){
		_biases[i] = initialBias;
	}
}

void MultilayerNetwork::BuildBincoder(int numInputs, int numLayers, const vector<int> neuronsPerLayer, const vector<ActivationFunction> activationSchema)
{
	BuildDeepNetwork(numInputs, numLayers, neuronsPerLayer, activationSchema, 0.0);

	//clamp all biases to zero; encoders don't have biases
	for(int i = 0; i < _biases.size(); i++){
		_biases[i] = 0.0;
	}

	AssignRandomWeights();
	PrintNetworkProperties();
	PrintWeights();
}

void MultilayerNetwork::PrintNetworkProperties()
{
	cout << "-----------Network-----------" << endl;

	cout << "Layers: " << _layers.size() << endl;

	cout << "Neurons per layer: ";
	for(int i = 0; i < _layers.size(); i++){
		cout << _layers[i].size() << "  ";
	}
	cout << endl;

	cout << "Biases: " << _biases.size() << endl;
	cout << "Bias values: ";
	for(int i = 0; i < _biases.size(); i++){
		cout << _biases[i] << "  ";
	}
	cout << endl;

	cout << "Layer activation functions: ";
	for(int i = 0; i < _layers.size(); i++){
		cout << _layers[i][0].GetActivationFunctionString(_layers[i][0].PhiFunction) << "  ";
	}
	cout << endl;
}

void MultilayerNetwork::BuildDeepMultiLabelNetwork(int numInputs, int numLayers, const vector<int> neuronsPerLayer, const vector<ActivationFunction> activationSchema)
{
	BuildDeepNetwork(numInputs, numLayers, neuronsPerLayer, activationSchema);
}

/*
A builder; this could be moved to some builder class

@numInputs: The dimension of the input data, such as 2 or 3-d. The network neurons will be configured
with numInputs + 1 inputs, where the additional input is for the bias.
@numInputs: number of inputs (attributes)
@numLayers: Number of layers. This should always be two, but its fun to experiment...
@numHiddenUnits: Number of hidden units in the hidden layer
@numOutputUnits: Nmber of output units
*/
void MultilayerNetwork::BuildNet(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits)
{
	int i, j, l, numLayerInputs;

	//clear any existing model
	Clear();

	//init the layers
	_layers.resize(numLayers);

	//lay out the hidden units
	for(i = 0; i < (numLayers-1); i++){
		//set number of inputs to either the number of network inputs, of the number of neurons in the previous layer (for successive hidden layers)
		if(i == 0){
			numLayerInputs = numInputs;
		}
		else{
			numLayerInputs = _layers[i-1].size();
		}

		_layers[i].resize(numHiddenUnits, Neuron(numLayerInputs + 1, TANH) );  //plus one for the biases
		_nullifyLayer(_layers[i]);
	}

	//set up the output units
	_layers[_layers.size()-1].resize(numOutputUnits, Neuron(_layers[_layers.size()-2].size() + 1, TANH) ); //plus one for the biases
	_nullifyLayer(_layers[_layers.size()-1]);

	//connect the inter-layer connections
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
	for(l = _layers.size()-1; l >= 0; l--){
		//init the 0th input of every neuron in this layer to the bias for the layer
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].Inputs[0] = &_biases[l];
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
	AssignRandomWeights(1.0,0.0);
}

/*
Prints weights of the entire network.
*/
void MultilayerNetwork::PrintWeights()
{
	int i, j, l;
	
	//print each layer
	for(l = 0; l < _layers.size(); l++){
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
When written to file, the network weights will be written out in layers, formatted as:
	numLayers=2
	numInputs=3
	hiddenUnitsPerLayer=8
	numOutputUnits=1
	hiddenLayerFunction=TANH
	outputFunction=LINEAR
	LAYER_1_WEIGHTS
	1 2 3
	2 3 1
	...
	LAYER_2_WEIGHTS
	3 4 5
	1 0 4
	...

TODO: A neural network is really just a graph. It might be easier to use graph serialization libs
or standards, eg graphml.
*/
void MultilayerNetwork::SaveNetwork(const string& path)
{
	int i, j, l;
	ofstream fileStream;

	fileStream.open(path);
	
	//write global structural info
	fileStream << "numLayers=" << _layers.size() << endl;
	fileStream << "numInputs=" << (_layers[0][0].Inputs.size() - 1) << endl; // minus one for the bias
	fileStream << "hiddenUnitsPerLayer=" << _layers[0].size() << endl;
	fileStream << "numOutputUnits=" << _layers[_layers.size()-1].size() << endl;
	fileStream << "hiddenLayerFunction=" << Neuron::GetActivationFunctionString(_layers[0][0].PhiFunction) << endl;
	fileStream << "outputFunction=" << Neuron::GetActivationFunctionString(_layers[_layers.size()-1][0].PhiFunction) << endl;

	//iterate the layers
	for(l = 0; l < _layers.size(); l++){
		//iterate each neuron in this layer
		fileStream << l << "_LAYER_WEIGHTS" << endl;
		for(i = 0; i < _layers[l].size(); i++){
			//output this neuron's weights
			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				fileStream << _layers[l][i].Weights[j].w;
				if(j < _layers[l][i].Weights.size() - 1)
					fileStream << ",";
				else
					fileStream << endl;
			}
		}
	}

	fileStream.close();
}

/*
Given a string of the form "numLayers=5", returns 5 as a double. This func doesn't
care about the param string value.
*/
double MultilayerNetwork::_getParamVal(const string& param)
{
	double val;

	cout << "Param: " << param << endl;
	val = stod(param.substr(param.find('=')+1, param.length() - (param.find('=')+1)));
	cout << "Val=" << val << endl;

	return val;
}

/*
This shouldn't be used, but is useful for hacking. Some problems inherently give rise to divergent
network training, whereby weights/outputs grow to +/-infinity or other huge numbers.

This allow resetting the network and starting over when such a condition is reached.
*/
void MultilayerNetwork::ResetWeights()
{

}


/*
Given some file of saved weights formatted as by SaveWeights(), this
deserializes them back into the network. Note this does not set eta, momentum,
etc., which is the client's responsibility.

This function is suicidal. If the input format is wrong, unordered, or values are invalid, die and go to heaven.
*/
void MultilayerNetwork::ReadNetwork(const string& path)
{
	int i, j, l;
	int numLayers, numHiddenUnits, numOutputUnits, numInputs;
	ActivationFunction hiddenLayerFunction, outputFunction;
	vector<double> temp;
	string input, param;
	ifstream ifile(path, ios::in);

	//parse numLayers param
	ifile >> input;
	numLayers = _getParamVal(input);
	//parse the numInputs
	ifile >> input;
	numInputs = _getParamVal(input);
	//parse the numHiddenUnitsPerLayer
	ifile >> input;
	numHiddenUnits = _getParamVal(input);
	//parse the numOutputs
	ifile >> input;
	numOutputUnits = _getParamVal(input);
	//parse the hiddenLayerFunction
	ifile >> input;
	hiddenLayerFunction = Neuron::GetActivationFunction(input.substr(input.find('=')+1, input.length() - (input.find('=')+1)));
	//parse the outputLayerFunction
	ifile >> input;
	outputFunction = Neuron::GetActivationFunction(input.substr(input.find('=')+1, input.length() - (input.find('=')+1)));

	//build the network
	Clear();
	BuildNet(numLayers, numInputs, numHiddenUnits, numOutputUnits);
	SetHiddenLayerFunction(hiddenLayerFunction);
	SetOutputLayerFunction(outputFunction);

	temp.resize( max(max(numOutputUnits,numHiddenUnits),numInputs) + 1); //plus one for a bias, if any

	//initialize the network weights; clearly this directly depends on the number of layers being initialized correctly
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			ifile >> input;
			//skip line, if this is the header line
			if(input.find("LAYER_WEIGHTS") != string::npos){
				ifile >> input;
			}

			_parseCsvFloats(input,temp);
			if(temp.size() < _layers[l][i].Weights.size()){
				cout << "ERROR dim-temp=" << temp.size() << " < dim-weights=" << _layers[l][i].Weights.size() << endl;
			}

			for(j = 0; j < _layers[l][i].Weights.size(); j++){
				_layers[l][i].Weights[j].w = temp[j];
				_layers[l][i].Weights[j].dw = 0;
			}
		}
	}

	ifile.close();
}

void MultilayerNetwork::_parseCsvFloats(string& input, vector<double>& vals)
{
	vector<string> temp;

	Tokenize(input,',',temp);

	if(vals.size() < temp.size()){
		//in the context of reading in a neural net, i just want to see this warning
		cout << "WARN passed double vector < dimension of csv row; check code" << endl;
		vals.resize(temp.size());
	}

	for(int i = 0; i < temp.size(); i++){
		vals[i] = stod(temp[i]);
	}
}

/*
There are specific strategies for initializing the weights (see Haykin). Here they are just init'ed with random numbers.
*/
void MultilayerNetwork::AssignRandomWeights(double high, double low)
{
	int i, j, l;
	
	//srand(time(NULL));
	
	//TODO: change the assignment to a zero-mean Gaussian, or other init per lit recommendations.
	for(l = 0; l < _layers.size(); l++){
		for(i = 0; i < _layers[l].size(); i++){
			_layers[l][i].AssignRandomWeights(1.0,0.0);
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

bool MultilayerNetwork::_isnormal(double val, bool allowZero)
{
	return std::isnormal(val) || (allowZero && val == 0.0);
}

//Validates output of network is not nan, inf, etc. This is critical so these
//values aren't backpropagated, which destroys the learned weights.
bool MultilayerNetwork::IsOutputNormal()
{
	vector<Neuron>& finalLayer = _layers[_layers.size()-1];
	for(int i = 0; i < finalLayer.size(); i++){
		if(!_isnormal(finalLayer[i].Output)){
			return false;
		}
	}

	return true;
}

/*
For the bincoder, the inputs are the target outputs

Not yet sure how to write the error function borrowed from word2vec...

void MultilayerNetwork::BincoderBackprop(const vector<double>& inputs)
{
	int i, j, l;
	double sum;

	//prevent nans, inf, etc from being backpropagated
	if(!IsOutputNormal()){
		cout << "ERROR one or more outputs abnormal, ABORTING BACKPROP" << endl;
		return;
	}

	//calculate the final-layer deltas, which are just the signal-prime values
	vector<Neuron>& finalLayer = _layers[_layers.size()-1];
	for(i = 0; i < finalLayer.size(); i++){
		//in Duda, this update is: delta = f'(net) * (t_k - z_k)
		finalLayer[i].Delta = finalLayer[i].PhiPrime() * (inputs[i] - finalLayer[i].Output);
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
*/

/*
Given that Classify() has been called for an example, this backpropagates the error given
by that single example. Hence this function is stateful, in that it assumes the network
outputs have been driven by a specific example. The reason this is public is for clients
that do online learning, like in approximate Q-learning.

@allowZero: Whether or not to treat zero as an abnormal target value
*/
void MultilayerNetwork::BackpropagateError(const vector<double>& inputs, const double target, bool allowZero)
{
	int i, j, l;
	double sum;

	//prevent nans, inf, etc from being backpropagated
	if(!_isnormal(target, allowZero) || !IsOutputNormal()){
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
Backprop for a binary-encoder/multilabel network with multihot binary output vector.
For binary-encoder, @inputs are also the target outputs.
*/
void MultilayerNetwork::BincoderBackpropagateError(const vector<double>& inputs, bool allowZero)
{
	int i, j, l;
	double sum;

	//prevent nans, inf, etc from being backpropagated
	if(!IsOutputNormal()){
		cout << "ERROR outputs abnormal, ABORTING BACKPROP" << endl;
		return;
	}

	//calculate the final-layer deltas, which are just the signal-prime values
	vector<Neuron>& finalLayer = _layers[_layers.size()-1];
	for(i = 0; i < finalLayer.size(); i++){
		//in Duda, this update is: delta = f'(net) * (t_k - z_k)
		finalLayer[i].Delta = finalLayer[i].PhiPrime() * (inputs[i] - finalLayer[i].Output);
		//cout << "delta " << finalLayer[i].Delta << endl;
	}

	//cout << "final layer size: " << finalLayer.size() << endl;
	//sleep(2);

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
void MultilayerNetwork::UpdateWeights(const vector<double>& inputs)
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

				//the weight update
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

				//subtract the regularization term (0.0 if unused) after all updates are done
				if(_weightDecayRate != 0.0){
					_layers[l][i].Weights[j].w *= (1.0 - _weightDecayRate);
				}
				//printf("%f\n",_layers[l][i].Weights[j]);
			}
		}
	}
}

/*
Classifies some portion of a dataset and evaluates the raw hamming error, just
for a raw test evaluation to see if a bincoding algorithm is working at all.

This currently requires the output is TANH (-1 for 0, +1 for 1), and applies
the function ceil(output) to each neuron to calculate the bits. So, an output
of -0.999 would evaluate to 0, 0.6 to 1, clamping the real outputs to actual bit values.
*/
void MultilayerNetwork::BincoderTest(const vector<vector<double> >& dataset)
{
	hammingErrors = 0.0; //all bits not matching

	for(int i = 0; i < dataset.size(); i++){
		sample = dataset[i];

		Classify(sample);
		for(int j = 0; j < sample.size(); j++){
			predictedBit = (int)ceil( _layers.back()[j].Output );
			actualBit = (int)sample[j];
			cout << "Predicted: " << predictedBit << "  Actual: " << actualBit << endl;
			if(predictedBit != actualBit){
				hammingErrors += 1;
			}
		}
	}

	cout << "Hamming errors: " << hammingError << endl;
}


/*

@dataset: A list of vectors, each of which is a sequence of 1's and 0's.
*/
void MultilayerNetwork::BincoderTrain(const vector<vector<double> >& dataset, double eta, double momentum, int maxIterations)
{
	bool done = false;
	string dummy;
	int minIteration, ringIndex;
	double convergence, input, netError, minError;
	vector<double> errorHistory;
	
	//sliding error-window for viewing long term error changes
	errorHistory.resize(1000, 0.0);
	ringIndex = 0;
	
	iterations = 0;
	netError = 1;
	minError = 10000000;
	convergence = 0;

	//initialize the weights to random values
	InitializeWeights();

	SetEta(eta);
	SetMomentum(momentum); //reportedly a good value is 0.5 (see Haykin)
	SetWeightDecay(0.01);

	//while(netError > convergenceThreshold){
	while(iterations < 500000 && !done){
		
		//randomly choose an example
		const vector<double>& example = dataset[ iterations % dataset.size() ];
		//learns from a single example: drives the network outputs, backprops, and updates the weights
		BincoderBackprop(example);
		
		//track error info
		netError = GetNetError();
		//cout << "error: " << netError << endl;
		//save this error level
		errorHistory[ringIndex] = abs(netError);
		ringIndex = (ringIndex + 1) % (int)errorHistory.size();
		
		iterations++;

		if(iterations % 10 == 9){
			//average the error of the last k examples learned
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

			/*
			PrintWeights();
			cout << "Continue? Enter 1 to end training: " << flush;
			cin >> dummy;
			done = dummy[0] == '1';
			*/
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
void MultilayerNetwork::StochasticBatchTrain(const vector<vector<double> >& dataset, double eta, double momentum)
{
	bool done = false;
	string dummy;
	int iterations, minIteration, ringIndex;
	double convergence, input, netError, minError;
	vector<double> errorHistory;
	
	//initialize the weights to random values
	InitializeWeights();
	
	//sliding error-window for viewing long term error changes
	errorHistory.resize(1000, 0.0);
	ringIndex = 0;
	
	iterations = 0;
	netError = 1;
	minError = 10000000;
	convergence = 0;
	SetEta(eta);
	SetMomentum(momentum); //reportedly a good value is 0.5 (see Haykin)
	SetWeightDecay(0.0001);

	//while(netError > convergenceThreshold){
	while(iterations < 500000 && !done){

		/*
		//serial/deserialization testing
		if(iterations == 1000){
			string path = "network.txt";
			cout << "Writing network to " << path << endl;
			SaveNetwork(path);
			Clear();
			ReadNetwork(path);
			SaveNetwork("network2.txt");
			cin >> path;
		}
		*/
		
		
		//randomly choose an example
		const vector<double>& example = dataset[ iterations % dataset.size() ];
		//learns from a single example: drives the network outputs, backprops, and updates the weights
		Backpropagate(example);
		
		//track error info
		netError = GetNetError();
		//save this error level
		errorHistory[ringIndex] = abs(netError);
		ringIndex = (ringIndex + 1) % (int)errorHistory.size();
		
		iterations++;

		if(iterations % 1000 == 999){
			//average the error of the last k examples learned
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

			/*
			PrintWeights();
			cout << "Continue? Enter 1 to end training: " << flush;
			cin >> dummy;
			done = dummy[0] == '1';
			*/
		}
	}

	PrintWeights();
	cout << "Minimum avg error per 1000 examples: " << minError << " at iteration " << minIteration << endl;
}

/*
Anoterh wrapper for stochastic batch training. Its up to the client to initialize the network as desired before call. 
*/
void MultilayerNetwork::StochasticBatchTrain(vector<TrainingExample>& examples, const int iterations)
{
	//string dummy;
	int minIteration, ringIndex;
	double avgError, minError;
	vector<double> errorHistory;
	
	//sliding error-window for viewing long term error changes
	errorHistory.resize(500, 0.0);
	ringIndex = 0;
	minError = 10000000;

	for(int i = 0; i < iterations; i++){
		//Backprop a random example
		TrainingExample& te = examples[ rand() % examples.size() ];
		Classify(te.xs);
		BackpropagateError(te.xs, te.target);
		UpdateWeights(te.xs);
		
		//All the following is just error tracking and reporting
		errorHistory[ringIndex] = abs( GetNetError() );
		ringIndex = (ringIndex + 1) % (int)errorHistory.size();
		if(i % (int)errorHistory.size() == (int)(errorHistory.size() - 1)){
			//average the error of the last k examples learned
			avgError = 0.0;
			for(int j = 0; j < errorHistory.size(); j++){
				avgError += errorHistory[j];
				errorHistory[j] = 0;
			}
			avgError /= (double)errorHistory.size();
			//cout << "\rIteration " << iterations << " avg error: " << avgError << "                " << flush;
			cout << "Iteration " << i << " avg error: " << avgError << "                " << endl;
			if(avgError < minError){
				minError = avgError;
				minIteration = iterations;
			}

			/*
			PrintWeights();
			cout << "Min error: " << minError << "  iteration " << minIteration << endl;
			cout << "Continue? Enter 1 to end training: " << flush;
			cin >> dummy;
			done = dummy[0] == '1';
			*/
		}
	}
	
	PrintWeights();
	cout << "Minimum avg error per 1000 examples: " << minError << " at iteration " << minIteration << endl;
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
		UpdateWeights(example);
	}
}

//Multilabel error backprop for multi-hot binary vectors
//@example: A binary vector of 0's and 1's
void MultilayerNetwork::BincoderBackprop(const vector<double>& example)
{
	//generic classification
	Classify(example);
	//back propagate the error measure/gradient
	BincoderBackpropagateError(example,true);
	//Update the weights based on the backpropagated error values
	UpdateWeights(example);
}

bool MultilayerNetwork::IsValidExample(const vector<double>& example)
{
	for(int i = 0; i < example.size(); i++){
		if(!_isnormal(example[i]) && example[i] != 0){
			cout << "WARNING In MultilayerNetwork, isnormal() returned true for input example value example[" << i << "] of " << example.size() << " length vector" << endl;
			cout << "Example will not be backpropagated" << endl;
			cout << "Value: " << example[i] << endl;
			cout << FpClassify(example[i]) << endl;
			return false;
		}
	}
	return true;
}

/*
Assuming Backpropagate() has been called after the network has been driven by some example, this
sums across the error across all output nodes.

In the future this could be factored or paramterized somehow to support error measures for
different gradient definitions.
*/
double MultilayerNetwork::GetNetError()
{
	double netError = 0.0;

	for(int i = 0; i < _layers.back().size(); i++){
		netError += _layers.back()[i].Delta;
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
			//cout << _layers.size() << " " << l << " " << i << " output: " << _layers[l][i].Output << endl;
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
		if(!_isnormal(_layers.back()[i].Output)){
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


//Given a line, tokenize it using delim, storing the tokens in output
void MultilayerNetwork::Tokenize(const string &s, char delim, vector<string> &tokens)
{
	stringstream ss(s);
	string temp;

	//clear any existing tokens
	tokens.clear();

	while (getline(ss, temp, delim)) {
		tokens.push_back(temp);
	}
}

/*
Reads in a csv file containing examples in the form <val1,val2...,val-n,class-label>
Example format for 2d data: 2.334,5.276,-1
The class lable is expected to be some kind of numeric, 1, -1, a probability, or even an integer-class (for multiclasses).
This function doesn't need to know any of that.

Returns: nothing, stores output in vector<double>.
*/
void MultilayerNetwork::ReadCsvDataset(const string& path, vector<vector<double> >& output)
{
	int i;
	fstream dataFile;
	string line;
	vector<string> tokens;
	vector<double> temp;

	dataFile.open(path.c_str(),ios::in);
	if(!dataFile.is_open()){
		cout << "ERROR could not open dataset file: " << path << endl;
	}

	cout << "Building dataset..." << endl;
	//clear any existing data
	output.clear();
	
	while(getline(dataFile, line)){
		tokens.clear();
		Tokenize(line,',',tokens);
		//cout << "here4: " << tokens.size() << " tokens" << endl;
		//build a temp double vector containing the vals from the record, in double form
		temp.clear();
		for(i = 0; i < tokens.size(); i++){
			//cout << tokens[i] << endl;
			temp.push_back(std::stod(tokens[i]));
		}
		
		output.push_back(temp);
	}
	cout << "Dataset build complete. Read " << output.size() << " examples." << endl;
}


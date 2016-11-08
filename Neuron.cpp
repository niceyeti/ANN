#include "Neuron.hpp"

Neuron::Neuron(int numInputs, ActivationFunction methodType)
{
	srand(time(NULL));

	Weights.resize(numInputs);
	for(int i = 0; i < Weights.size(); i++){
		Weights[i].w = 0;
		Weights[i].dw = 0;
	}

	Inputs.resize(numInputs);
	for(int i = 0; i < Inputs.size(); i++){
		Inputs[i] = NULL;
	}

	PhiFunction = methodType;
	Output = 0;
	Delta = 0;
	Signal = 0;
}

Neuron::Neuron(const Neuron& lhs)
{
	if(this != &lhs){
		PhiFunction = lhs.PhiFunction;
		Output = lhs.Output;
		Delta = lhs.Delta;
		Signal = lhs.Signal;
		
		Weights.resize(lhs.Weights.size());
		//copy the weights
		for(int i = 0; i < lhs.Weights.size(); i++){
			Weights[i].w = lhs.Weights[i].w;
			Weights[i].dw = lhs.Weights[i].dw;
		}
		
		Inputs.resize(lhs.Inputs.size());
		for(int i = 0; i < lhs.Inputs.size(); i++){
			Inputs[i] = lhs.Inputs[i];
		}
	}
}

Neuron::~Neuron()
{
	Weights.clear();
	Inputs.clear();
}

/*
Trains the neuron according to the labelled dataset.
The vector of vectors is essentially a matrix, as many lit write-ups describe it.
Each vector is augmented with +1.0 or -1.0 in the last column to indicate binary class membership.

void Neuron::Train(vector<vector<double> >& dataset, int traingMethod)
{
	
	
}
*/

void Neuron::NullifyInputPtrs()
{
	for(int i = 0; i < Inputs.size(); i++){
		Inputs[i] = NULL;
	}
}

//All of the following functions can be found in the neural net literature.
double Neuron::Sigmoid(double expt)
{
	//these are accurate to many decimals, and abort huge exponentiations
	if(expt >= 20){
		return 0.0;
	}
	if(expt <= -20){
		return 1.0;
	}

	return 1.0 / (1.0 + exp(-expt));
}

//Returns first derivative of the sigmoid function
double Neuron::SigmoidPrime(double expt)
{
	return Sigmoid(expt) * (1.0 - Sigmoid(expt));
}

const char* Neuron::FpClassify(double x)
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

void Neuron::ValidateOutput()
{
	if(!std::isnormal(this->Output) && this->Output != 0.0){
		cout << "WARNING Neuron Output non-normal: " << FpClassify(Output) << endl; 
	}
}

void Neuron::ValidateSignal()
{
	if(!std::isnormal(this->Signal) && this->Signal != 0.0){
		cout << "WARNING Neuron Signal non-normal: " << FpClassify(Signal) << endl; 
	}
}

/*
The textbook version of Tanh is (exp(expt) - exp(-expt)) / (exp(expt) + exp(-expt)).
This can be simplified to the following. Wikipedia "Activation Function" has a nice overview
of popular activations and their derivatives.
*/
double Neuron::Tanh(double expt)
{
	//these are correct to 20 or so decimals
	if(expt <= -20.0){
		return -1.0;
	}
	if(expt >= 20.0){
		return 1.0;
	}

	return (2.0 / (1.0 + exp(-2.0*expt))) - 1.0;
}

//Returns first derivative of the tanh function
double Neuron::TanhPrime(double expt)
{
	return 1.0 - (Tanh(expt) * Tanh(expt));
}

/*
Calculates, sets and returns the Signal for this neuron. A 'signal' is almost always the
dot product of the weights and the inputs. This is stateful. The inputs must have been set,
and the weights assigned previously.

Precondition: Inputs and weights have been set.
*/
double Neuron::CalculateSignal()
{
	Signal = InnerProduct(Inputs, Weights);
	
	return Signal;
}

/*
Assigns random weights to the neuron, restricted to range [low,high].
*/
void Neuron::AssignRandomWeights(double high, double low)
{
	//srand(time(NULL));

	for(int i = 0; i < Weights.size(); i++){
		//inits random weights in range [-1.0,1.0]
		Weights[i].w = ((double)(rand() % 10000) / (double)10000.0) * (high - low) + low;
		Weights[i].dw = 0.0;
		cout << "weight " << i << " " << Weights[i].w << endl;
		//Weights[i].w = (((double)(rand() % 400)) / 100.0) - 1.0;
	}
}

void Neuron::Stimulate()
{
	CalculateSignal();
	ValidateSignal();
	Phi();
	ValidateOutput();
}

/*
The Signal of a neuron is just the dot product of its weights and the current inputs. The Output of abort
neuron is h(Signal), where h() is some smooth function like tanh, sigmoid, etc. Here, the prescribed
h(x) is applied to the Signal and returned;

Precondition: Signal has been set.
*/
double Neuron::Phi()
{
	//just map and call this neuron's activation function
	switch(PhiFunction){

		case TANH:
				Output = Tanh(Signal);
			break;

		case LOGISTIC:
				Output = Sigmoid(Signal);
			break;
			
		case LINEAR:
				Output = Signal;
			break;

		case SIGN:
				Output = (Signal >= 0) ? 1.0 : -1.0;
			break;
	
		default:
				cout << "ERROR unknown output type: " << (int)PhiFunction << endl;
			break;
	}
	
	return Output;
}

/*
Returns first derivative of phi-function, using current signal as input.
*/
double Neuron::PhiPrime()
{
	double result = 0;
	
	//just map and call this neuron's activation function
	switch(PhiFunction){
		//only these two functions are differentiable; the others aren't intended for neurons for which this function would be called
		case TANH:
				result = TanhPrime(Signal);
			break;
			
		case LOGISTIC:
				result = SigmoidPrime(Signal);
			break;

		case LINEAR:
				result = 1.0;
			break;
	
		default:
				cout << "ERROR unknown output type in PhiPrime(): " << (int)PhiFunction << endl;
			break;
	}

	return result;
}

ActivationFunction Neuron::GetActivationFunction(const string& function)
{
	ActivationFunction af = LOGISTIC;

	if(function == "TANH"){
		af = TANH;
	}
	else if(function == "LOGISTIC"){
		af = LOGISTIC;
	}
	else if(function == "LINEAR"){
		af = LINEAR;
	}
	else if(function == "SIGN"){
		af = SIGN;
	}
	else{
		cout << "Activation function unknown: " << function << "  returning LOGISTIC" << endl;
	}

	return af;
}

string Neuron::GetActivationFunctionString(ActivationFunction function)
{
	string funcStr;

	switch(function){
		case TANH:
			funcStr = "TANH";
		break;
		case LOGISTIC:
			funcStr = "LOGISTIC";
		break;
		case LINEAR:
			funcStr = "LINEAR";
		break;
		case SIGN:
			funcStr = "SIGN";
		break;
		default:
			funcStr = "UNKNOWN";
		break;
	}

	return funcStr;
}

double Neuron::InnerProduct(const vector<const double*>& inputs, const vector<Weight>& weights)
{
	double sum = 0.0;

	if(inputs.size() != weights.size()){
			cout << "ERROR neuron inputs and weight vector sizes unequal: weights = " << weights.size() << "  inputs=" << inputs.size() << endl;
	}

	for(int i = 0; i < inputs.size(); i++){
		sum += (*inputs[i]) * weights[i].w;
	}
	
	return sum;
}

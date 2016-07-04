#include "Neuron.hpp"

Neuron::Neuron(int numInputs, ActivationFunction methodType)
{
	Weights.resize(numInputs);
	Inputs.resize(numInputs);
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
			Weights[i] = lhs.Weights[i];
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
	return 1 / (1 + exp(-expt));
}

//Returns first derivative of the sigmoid function
double Neuron::SigmoidPrime(double expt)
{
	return Sigmoid(expt) * (1.0 - Sigmoid(expt));
}

double Neuron::Tanh(double expt)
{
	return (exp(expt) - exp(-expt)) / (exp(expt) + exp(-expt));
}

//Returns first derivate of the tanh function
double Neuron::TanhPrime(double expt)
{
	return 1.0 - pow(Tanh(expt),2.0);
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

void Neuron::Stimulate()
{
	CalculateSignal();
	Phi();
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
			
		case LINEAR:
				Output = Signal;
			break;

		case SIGN:
				Output = (Signal >= 0) ? 1.0 : -1.0;
			break;
			
		case LOGISTIC:
				Output = Sigmoid(Signal);
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
	
		default:
				cout << "ERROR unknown output type: " << (int)PhiFunction << endl;
			break;
	}

	return result;	
}

double Neuron::InnerProduct(const vector<const double*>& inputs, const vector<double>& weights)
{	
	if(inputs.size() != weights.size()){
			cout << "ERROR neuron inputs and weight vector sizes unequal: weights = " << weights.size() << "  inputs=" << inputs.size() << endl;
			exit(0);
	}

	double sum = 0.0;
	for(int i = 0; i < inputs.size(); i++){
		sum += (*inputs[i]) * weights[i];
	}
	
	return sum;
}

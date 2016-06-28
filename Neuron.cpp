#include "Neuron.hpp"

Neuron::Neuron(int numInputs, int methodType)
{
	Weights.resize(numInputs);
	Method = methodType;
	Output = 0;
}

Neuron::~Neuron()
{
	Weights.clear();
}

/*
Trains the neuron according to the labelled dataset.
The vector of vectors is essentially a matrix, as many lit write-ups describe it.
Each vector is augmented with +1.0 or -1.0 in the last column to indicate binary class membership.
*/
void Neuron::Train(vector<vector<double> >& dataset, int traingMethod)
{
	
	
}

//All of the following functions can be found in the neural net literature.
double Neuron::Sigmoid(double expt)
{
	return 1 / (1 + exp(-expt));
}

//Returns first derivative of the sigmoid function
double Neuron::SigmoidPrime(double expt)
{
	return Sigmoig(expt) * (1.0 - Sigmoid(expt));
}

double Neuron::Tanh(double expt)
{
	return (exp(expt) - exp(-expt)) / (exp(expt) - exp(-expt));
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
	Signal = 0.0
	for(int i = 0; i < _inputs.size(); i++){
		Signal += (_inputs[i] * _weights);
	}
	
	return Signal;
}

/*
The Signal of a neuron is just the dot product of its weights and the current inputs. The Output of abort
neuron is h(Signal), where h() is some smooth function like tanh, sigmoid, etc. Here, the prescribed
h(x) is applied to the Signal and returned;

Precondition: Signal has been set.
*/
double Neuron::CalculateOutput()
{
	//just map and call this neuron's activation function
	switch(ActivationType){
			case ActivationFunction.TANH:
					Output = Tanh(Signal);
				break;
				
			case ActivationFunction.LINEAR:
					Output = Signal;
				break;
			case ActivationFunction.SIGN:
					Output = (Signal >= 0) 1.0 : -1.0;
				break;
				
			case ActivationFunction.SIGMOID:
					Output = Sigmoid(Signal);
				break;
		
			default:
					cout << "ERROR unknown output type: " << ActivationType << endl;
				break;
	}
	
	return Output;
}




/*
double Neuron::Sigmoid(const vector<double>& inputs)
{
	return Sigmoid( InnerProduct(inputs, this->Weights) );
}

double Neuron::Sigmoid(const vector<double>& inputs, const vector<double>& weights)
{
	return Sigmoid( InnerProduct(inputs, weights) );
}
*/

double Neuron::InnerProduct(const vector<double>& inputs, const vector<double>& weights)
{	
	if(inputs.size() != weights.size()){
			cout << "ERROR neuron inputs and weight vector sizes unequal: weights = " << weights.size() << "  inputs=" << inputs.size() << endl;
			exit(0);
	}

	double sum = 0.0;	
	for(int i = 0; i < inputs.size(); i++){
		sum += inputs[i] * weights[i];
	}
	
	return sum;
}

/*
Simulates the neuron on the inputs. The output is both returned and also saved in Output.
*/
double Neuron::Simulate(vector<double>& inputs)
{
	double Output = 0;
	
	switch(methodType){
		case SIGMOID:
				Output = Sigmoid(inputs);
				break;
		
		case LINEAR:
				Output = Linear(inputs);
				break;
		
		default:
			cout << "ERROR GetOutput() method unknown: " << methodType << endl;
			exit(0);
			break;
	}
	
	return Output;
}
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

/*
An implementation of linear discriminant functions. Hopefully I can keep these useful for both linear discriminant/regression
applications, as well as for neural network components.

*/


enum ActivationFunction {TANH, LOGISTIC, LINEAR, SIGN};

class Neuron{
	public:
		Neuron(const Neuron& lhs);
		Neuron(int numInputs, ActivationFunction methodType);
		~Neuron();
		
		vector<double> Weights;
		vector<const double*> Inputs; //inputs are pointer-type, so that a single input may be demuxed to multiple neurons
		ActivationFunction PhiFunction;
		double Signal; // the weighted sum of inputs (the dot product w * x before being passed through the signal function, theta)
		double Output;
		double Delta; //assigned by the backpropagation algorithm.
		void NullifyInputPtrs();
		void Stimulate();
		double CalculateSignal();
		double Phi();
		double PhiPrime();
		double InnerProduct(const vector<const double*>& inputs, const vector<double>& weights);
		//All of the following functions can be found in the neural net literature.
		double Sigmoid(double expt);
		//Returns first derivative of the sigmoid function
		double SigmoidPrime(double expt);
		double Tanh(double expt);
		//Returns first derivative of the tanh function
		double TanhPrime(double expt);
};

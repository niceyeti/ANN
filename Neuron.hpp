#include <cmath>

/*
An implementation of linear discriminant functions. Hopefully I can keep these useful for both linear discriminant/regression
applications, as well as for neural network components.

*/


enum class ActivationFunction{TANH, LOGISTIC, LINEAR, SIGN };

class Neuron{
	public:
		vector<double> Weights;
		vector<double> Inputs;
		//vector<double> Inputs; removed this from the neuron, since it makes more sense to have the inputs be provided by an external source, such as other nn outputs
		enum ActivationFunction ActivationType;
		double Signal; // the weighted sum of inputs (the dot product w * x before being passed through the signal function, theta)
		double Output;
		double Delta; //assigned by the backpropagation algorithm.
		Neuron(int numInputs, int methodType=SIGMOID);
		~Neuron();

		double Simulate(vector<double>& inputs);
		double Sigmoid(const vector<double>& inputs, const vector<double>& weights);
		double Sigmoid(const vector<double>& inputs);
		double Sigmoid(double expt);
		void Train(vector<vector<double> >& dataset, int traingMethod);
};

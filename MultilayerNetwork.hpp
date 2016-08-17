#ifndef MULTILAYER_NETWORK_HPP
#define MULTILAYER_NETWORK_HPP

#include "Neuron.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <cfloat>


using namespace std;

/*
Implements a basic, Multilayer ANN with sigmoidal units.

This is the most generic form of ANN, and is based directly off of the BackPropagation methods mentioned in
Duda "Pattern Classification" 6.3. It isn' intended to be efficient or extensible, just a proto to learn the basics.
Also see Mustafa's CalTech Multi-Layer NN lecture in the Learning from Data series on youtube.


TODO: Neural nets are amenable to some sort of Builder or factory pattern, if this code is reused.

TODO: If any inputs are nan, inf, or other abnormal floats, the network is basically shot once these are
backpropagated through the weights, which then become abnormal themselves. Fix this behavior.
*/

/*
This is just a single-purpose architecture, or learning about ANN's.

As far as software patterns, ANN's are highly suited to using some builder (Builder, Factory, etc.) to construct and return ANN's with different
architectures (numbers of hidden layers, num hidden units/outputs, etc.), under a single api/interface.
*/
class MultilayerNetwork{
	private:
		double _eta;
		vector<double> _biases;
		vector<vector<Neuron> > _layers;
		//eta performs best with some decaying value updates
		void _nullifyLayer(vector<Neuron>& layer);
		void _assignRandomWeights();
	public:
		bool IsOutputNormal();
		void SetHiddenLayerFunction(ActivationFunction functionType);
		void SetOutputLayerFunction(ActivationFunction functionType);
		void SetEta(double newEta);
		void InitializeWeights();
		const vector<Neuron>& GetOutputs(); //client gets the output by reading the output layer neurons' outputs
		void PrintWeights();
		double GetNetError();
		void ValidateOutputs();
		const char* FpClassify(double x);
		bool IsValidExample(const vector<double>& example);
		void BuildNet(int numInputs, int numLayers, int numHiddenUnits, int numOutputUnits);
		void Classify(const vector<double>& inputs);
		void UpdateWeights(const vector<double>& inputs, double target);
		void BackpropagateError(const vector<double>& inputs, double target);
		void Backpropagate(const vector<double>& example);
		void BatchTrain(const vector<vector<double> >& dataset);
		void Test(const string& outputPath, vector<vector<double> >& testSet);
		MultilayerNetwork();
		MultilayerNetwork(int numInputs, int numLayers, int numHiddenUnits, int numOutputUnits);
		~MultilayerNetwork();
		void Clear();
};

#endif
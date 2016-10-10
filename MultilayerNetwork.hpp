#ifndef MULTILAYER_NETWORK_HPP
#define MULTILAYER_NETWORK_HPP

#include "Neuron.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <fstream>

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
		bool _useRegularizer;
		double _eta;
		double _weightDecayRate; //l2 regularizer
		double _momentum;
		vector<double> _biases;
		vector<vector<Neuron> > _layers;
		//eta performs best with some decaying value updates
		void _nullifyLayer(vector<Neuron>& layer);
		void _tokenize(const string &s, char delim, vector<string> &tokens);
		void _parseCsvFloats(string& input, vector<double>& vals);
		double _getParamVal(const string& param);
	public:
		string Name;
		void SetWeightDecay(double decayRate);
		void ReadNetwork(const string& path);
		void SaveNetwork(const string& path);
		void AssignRandomWeights();
		void ReadCsvDataset(const string& path, vector<vector<double> >& output);
		void SetMomentum(double momentum);
		bool IsOutputNormal();
		void SetHiddenLayerFunction(ActivationFunction functionType, int layer=0);
		void SetOutputLayerFunction(ActivationFunction functionType);
		void SetEta(double newEta);
		void InitializeWeights();
		const vector<Neuron>& GetOutputs(); //client gets the output by reading the output layer neurons' outputs
		void PrintWeights();
		void ResetWeights();
		void ReadWeights(const string& path);
		void SaveWeights(const string& path);
		double GetNetError();
		void ValidateOutputs();
		const char* FpClassify(double x);
		bool IsValidExample(const vector<double>& example);
		void BuildNet(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits);
		void Classify(const vector<double>& inputs);
		void UpdateWeights(const vector<double>& inputs, double target);
		void BackpropagateError(const vector<double>& inputs, double target);
		void Backpropagate(const vector<double>& example);
		void StochasticBatchTrain(const vector<vector<double> >& dataset, double eta, double momentum);
		void Test(const string& outputPath, vector<vector<double> >& testSet);
		MultilayerNetwork();
		MultilayerNetwork(int numLayers, int numInputs, int numHiddenUnits, int numOutputUnits);
		~MultilayerNetwork();
		void Clear();
};

#endif

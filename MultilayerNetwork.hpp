#include "Neuron.hpp"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

/*
Implements a basic, Multilayer ANN with sigmoidal units.

This is the most generic form of ANN, and is based directly off of the BackPropagation methods mentioned in
Duda "Pattern Classification" 6.3. It isn' intended to be efficient or extensible, just a proto to learn the basics.
Also see Mustafa's CalTech Multi-Layer NN lecture in the Learning from Data series on youtube.


TODO: Neural nets are amenable to some sort of Builder or factory pattern, if this code is reused.
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
		void _buildTwoLayerNet(int numInputs, int numHiddenUnits, int numOutputUnits);
		void _assignRandomWeights();
	public: 
		const vector<Neuron>& GetOutput(); //client gets the output by reading the output layer neurons' outputs
		void PrintWeights();
		double GetNetError();
		void Classify(const vector<double>& example);
		void UpdateWeights(const vector<double>& example);
		void BackpropagateError(const vector<double>& example);
		void Backpropagate(const vector<double>& example);
		void BatchTrain(const vector<vector<double> >& dataset);
		void Test(const string& outputPath, vector<vector<double> >& testSet);
		MultilayerNetwork(int numInputs, int numHiddenUnits, int numOutputUnits);
		~MultilayerNetwork();
		void Clear();
};

#include "Neuron.hpp"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

/*
Implements a basic, multilayer ANN with sigmoidal units.

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
class MultiLayerNetwork{
	private:
		double _eta;
		vector<double> _biases;
		vector<vector<Neuron> > _layers;
		//eta performs best with some decaying value updates
		void _nullifyLayer(vector<Neuron>& layer);
		void _buildTwoLayerNet(int numInputs, int numHiddenUnits, int numOutputUnits);
		void _assignRandomWeights();
	public: 
		void PrintWeights();
		void BackPropagation(const vector<vector<double> >& dataset, double eta);
		void Classify(const vector<double>& example);
		void Test(vector<vector<double> >& dataset);
		MultiLayerNetwork(int numInputs, int numHiddenUnits, int numOutputUnits);
		~MultiLayerNetwork();
		void Clear();
};

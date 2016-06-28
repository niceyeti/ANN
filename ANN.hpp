/*
Implements a basic, multilayer ANN with sigmoidal units.

This is the most generic form of ANN, and is based directly off of the BackPropagation methods mentioned in
Duda "Pattern Classification" 6.3. It isn' intended to be efficient or extensible, just a proto to learn the basics.
Also see Mustafa's CalTech Multi-Layer NN lecture in the Learning from Data series on youtube.

*/

/*
This is just a single-purpose architecture, or learning about ANN's.

As far as software patterns, ANN's are highly suited to using some builder (Builder, Factory, etc.) to construct and return ANN's with different
architectures (numbers of hidden layers, num hidden units/outputs, etc.), under a single api/interface.
*/
class MultiLayerNetwork{
	private:
		vector<Neuron> > _hiddenUnits;
		vector<Neuron> _outputUnits;
		//eta performs best with some decaying value updates
		void _updateEta();
	public: 
		void ReadDataset(const string& path);
		double Eta;
		vector<double> Outputs;
		//single layer of hidden units for now;
		vector<double> HiddenOutputs;
		MultiLayerNet(int numHiddenUnits, int numOutputUnits);
		~MultiLayerNet();
		void BackPropagation(vector<vector<double> >& dataset);
		void Classify(vector<double>& example);
		void Test(vector<vector<double> >& dataset);
		void Train(const string& path, const double eta);
};

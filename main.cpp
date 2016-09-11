#include "MultilayerNetwork.hpp"

void usage()
{
	cout << "Usage: ./batch [path to dataset] [#layers] [output function] [hidden function] [#numInputs] [#hidden units] [#outputs] [eta] [momentum]" << endl;
	cout << "For 'function' parameters, valid values are: LOGISTIC, TANH, or LINEAR" << endl;
}

int main(int argc, char** argv)
{
	int numLayers, numHiddenUnits, numInputs, numOutputs;
	double momentum, eta;
	ActivationFunction outputFunc, hiddenFunc;
	string path, outputFunction, hiddenFunction;
	vector<vector<double> > dataset;
	
	if(argc < 10){
		cout << "Incorrect num parameters: " << argc << endl;
		usage();
		return 0;
	}

	path = argv[1];
	numLayers = stoi(argv[2]);
	outputFunction = argv[3];
	hiddenFunction = argv[4];
	numInputs = stoi(argv[5]);
	numHiddenUnits = stoi(argv[6]);
	numOutputs = stoi(argv[7]);
	eta = stod(string(argv[8]));
	momentum = stod(string(argv[9]));

	outputFunc = Neuron::GetActivationFunction(outputFunction);
	hiddenFunc = Neuron::GetActivationFunction(hiddenFunction);

	//build the network from the cmd line parameters
	MultilayerNetwork nn(numLayers, numInputs, numHiddenUnits, numOutputs);
	//set up the network output and hidden layer functions
	nn.SetOutputLayerFunction(outputFunc);
	nn.SetHiddenLayerFunction(hiddenFunc);
	//read the dataset
	nn.ReadCsvDataset(path, dataset);

	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	nn.BatchTrain(dataset, eta, momentum);

	/*
	dataset.clear();
	path = "./Data/test2d.csv";
	nn.ReadCsvDataset(path, dataset);
	path = "./Data/predictions2d.csv";
	nn.Test(path, dataset);
	*/	


	return 0;
}

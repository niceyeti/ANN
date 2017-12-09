#include "MultilayerNetwork.hpp"
#include <cstring>

/*
Main driver for some preliminary viability testing of deep boolean networks
Let the input dataset be a vector of boolean components and a single +/-1 output.
For process mining graph collections, the collection of boolean vectors represent adjacency matrices.
This is unusual, since the input size of the neural net is not fixed, but changes wrt the v^2, the number of 
possible directed edges in the network.
*/


bool GetActivationSeq(char* arg, vector<ActivationFunction>& activationFunctions)
{	
	bool success = true;
	activationFunctions.clear();
	
	for(int i = 0; i < strlen(arg); i++){
		switch((int)arg[i]){
			case 'O':
				activationFunctions.push_back(LOGISTIC);
				break;
			case 'T':
				activationFunctions.push_back(TANH);
				break;
			case 'L':
				activationFunctions.push_back(LINEAR);
				break;
			case ',':
				//do nothing; skip
				break;
			default:
				cout << "ERROR function symbol not found: " << arg[i] << endl;
				success = false;
				break;
		}
	}

	return success;
}

void usage()
{
	cout << "Usage: ./batch [path to dataset] [#layers] [#numInputs] [hidden layer size sequence: 10,3,1] [function csv sequence: O,T,L ]  [eta] [momentum]" << endl;
	cout << "For 'function' parameters, valid values are: l(O)gistic, (T)anh, or (L)inear" << endl;
	cout << "Function csv sequence: the sequence of functions used by each hidden layer" << endl;
	cout << "Hidden layer sequence: a csv integer sequence signifying the number of neurons in each layer, where the first item is the size of the first hidden layer, etc" << endl;
}

int main(int argc, char** argv)
{
	int numLayers, numInputs;
	vector<int> neuronsPerLayer;
	vector<ActivationFunction> activationFunctions;
	double momentum, eta;
	string path, outputFunction, hiddenFunction;
	vector<vector<double> > dataset;
	MultilayerNetwork nn;	
	vector<string> tokens;

	if(argc < 8){
		cout << "Incorrect num parameters: " << argc << endl;
		usage();
		return 0;
	}

	path = argv[1];
	numLayers = stoi(argv[2]);
	numInputs = stoi(argv[3]);
	
	nn.Tokenize(string(argv[4]), ',', tokens);
	for(int i = 0; i < tokens.size(); i++){
		neuronsPerLayer.push_back(stoi(tokens[i]));
		cout << "Neurons in layer " << i << ": " << neuronsPerLayer[i] << endl;
	}

	if((int)neuronsPerLayer.size() != numLayers){
		cout << "Incorrect layer size sequence passed: " << argv[4] << ". Must be same as passed numLayers: " << numLayers << endl;
		usage();
		return 1;
	}
	
	if(!GetActivationSeq(argv[5], activationFunctions) || (int)activationFunctions.size() != numLayers){
		cout << "Incorrect function sequence passed: " << argv[5] << endl;
		usage();
		return 1;
	}
	
	eta = stod(string(argv[6]));
	momentum = stod(string(argv[7]));

	//build the network from the cmd line parameters
	nn.BuildBincoder(numInputs, numLayers, neuronsPerLayer, activationFunctions);
	//set up the network output and hidden layer functions
	//read the dataset
	nn.ReadCsvDataset(path, dataset);

	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	int iterations = 20;

	while(1){
		nn.BincoderTrain(dataset, eta, momentum, iterations);
		nn.BincoderTest(dataset);
	}


	/*
	dataset.clear();
	path = "./Data/test2d.csv";
	nn.ReadCsvDataset(path, dataset);
	path = "./Data/predictions2d.csv";
	nn.Test(path, dataset);
	*/	


	return 0;
}

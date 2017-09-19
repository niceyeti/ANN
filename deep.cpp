#include "MultilayerNetwork.hpp"

/*
Main driver for some preliminary viability testing of deep boolean networks
Let the input dataset be a vector of boolean components and a single +/-1 output.
For process mining graph collections, the collection of boolean vectors represent adjacency matrices.
This is unusual, since the input size of the neural net is not fixed, but changes wrt the v^2, the number of 
possible directed edges in the network.
*/

void GetLayerSizeSeq(char* arg, vector<int>& neuronsPerLayer)
{
	char *ptrs[50] = {NULL};
	char* temp = NULL;
	int j = 0;
	
	neuronsPerLayer.clear();
	
	temp = arg;
	for(int i = 0; i < strlen(arg); i++){
		if(arg[i] == ','){
			ptrs[j] = temp;
			j++;
			arg[i] = '\0';
		}
	}
	
	for(int i = 0; i < j; i++){
		neuronsPerLayer.push_back(stoi(ptrs[i]));
	}
}

void GetActivationSeq(char* arg, vector<ActivationFunction>& activationFunctions)
{	
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
				break;
		}
	}
}

void usage()
{
	cout << "Usage: ./batch [path to dataset] [#layers] [#numInputs] [hidden layer size sequence: 10,3,1] [function csv sequence: O,T,L ]  [eta] [momentum]" << endl;
	cout << "For 'function' parameters, valid values are: LOGISTIC (O), TANH (T), or LINEAR (L)" << endl;
	cout << "Function csv sequence: the sequence of functions used by each  hidden layer" << endl;
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
	
	if(argc < 8){
		cout << "Incorrect num parameters: " << argc << endl;
		usage();
		return 0;
	}

	path = argv[1];
	numLayers = stoi(argv[2]);
	numInputs = stoi(argv[3]);
	
	if(GetLayerSizeSeq(argv[4]), neuronsPerLayer){
		cout << "Incorrect layer size sequence passed: " << argv[4] << endl;
		usage()
		return 1;
	}
	
	if(GetActivationSeq(argv[5], activationFunctions) < 0){
		cout << "Incorrect function sequence passed: " << argv[5] << endl;
		usage();
		return 1;
	}
	
	eta = stod(string(argv[6]));
	momentum = stod(string(argv[7]));

	//build the network from the cmd line parameters
	MultilayerNetwork nn();
	nn.BuildDeepNetwork(numInputs, numLayers, neuronsPerLayer, activationFunctions);
	//set up the network output and hidden layer functions
	//read the dataset
	nn.ReadCsvDataset(path, dataset);

	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	nn.StochasticBatchTrain(dataset, eta, momentum);

	/*
	dataset.clear();
	path = "./Data/test2d.csv";
	nn.ReadCsvDataset(path, dataset);
	path = "./Data/predictions2d.csv";
	nn.Test(path, dataset);
	*/	


	return 0;
}

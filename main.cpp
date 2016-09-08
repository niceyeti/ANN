#include "MultilayerNetwork.hpp"

void usage()
{
	cout << "Usage: ./testAnn [path to dataset] [function: LOGISTIC, TANH, or LINEAR]" << endl;
}


int main(int argc, char** argv)
{
	ActivationFunction output;
	string path, function;
	MultilayerNetwork nn(2, 3, 25, 1);
	vector<vector<double> > dataset;
	
	if(argc < 3){
		cout << "Incorrect num parameters" << endl;
		usage();
		return 1;
	}

	path = argv[1];
	function = argv[2];

	if(function == "TANH"){
		output = TANH;
	}
	else if(function == "LINEAR"){
		output = LINEAR;
	}
	else if(function == "LOGISTIC"){
		output = LOGISTIC;
	}
	else{
		cout << "Function not found: " << function << endl;
		usage();
		return 1;
	}


	nn.SetOutputLayerFunction(output);
	nn.ReadCsvDataset(path, dataset);

	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	nn.BatchTrain(dataset,0.05,0.5);

	dataset.clear();
	path = "./Data/test2d.csv";
	nn.ReadCsvDataset(path, dataset);
	path = "./Data/predictions2d.csv";
	nn.Test(path, dataset);
	
	return 0;
}

#include "MultilayerNetwork.hpp"
#include "Util.hpp"

int main(int argc, char** argv)
{
	string path;
	MultilayerNetwork nn(2, 2, 8, 1);
	vector<vector<double> > dataset;
	
	if(argc < 2){
		cout << "No dataset path passed." << endl;
		return 1;
	}

	path = argv[1];

	nn.SetOutputLayerFunction(TANH);
	nn.ReadCsvDataset(path, dataset);

	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	nn.BatchTrain(dataset,0.1,0.0);

	dataset.clear();
	path = "./Data/test2d.csv";
	nn.ReadCsvDataset(path, dataset);
	path = "./Data/predictions2d.csv";
	nn.Test(path, dataset);
	
	return 0;
}

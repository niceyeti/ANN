#include "ANN.hpp"
#include "Util.hpp"

int main(int argc, char** argv)
{
	string path;
	MultiLayerNetwork ann(2, 8, 1);
	double eta = 0.1;
	vector<vector<double> > dataset;
	
	path = "./Data/training2d.csv";
	readCsv(path, dataset);
	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	ann.BackPropagation(dataset);

	dataset.clear();
	path = "./Data/test2d.csv";
	readCsv(path, dataset);
	path = "./Data/predictions2d.csv";
	ann.Test(path, dataset);
	
	return 0;
}
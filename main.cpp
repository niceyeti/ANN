#include "ANN.hpp"
#include "Util.hpp"

int main(int argc, char** argv)
{
	string path;
	MultiLayerNetwork ann(2, 8, 1);
	double eta = 0.1;
	vector<vector<double> > dataset;
	
	path = "./Data/data_2d.csv";
	readCsv(path, dataset);
	//for(int i = 0; i < dataset.size(); i++){
	//	for(int j = 0; j < dataset[i].size(); j++){
	//		cout << dataset[i][j] << " ";
	//	}
	//	cout << endl;
	//}
	ann.BackPropagation(dataset, 0.1);

	return 0;
}

#include "ANN.hpp"
#include "Util.hpp"

int main(int argc, char** argv)
{
	string path;
	MultiLayerNetwork ann(2, 8, 1);
	double eta;
	vector<vector<double> > dataset;
	
	path = "data_2d.csv";
	readCsv(path, dataset);
	//ann.BackPropagation(dataset);

	return 0;
}

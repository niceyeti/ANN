#include "Util.hpp"

//Given a line, tokenize it using delim, storing the tokens in output
void tokenize(const string &s, char delim, vector<string> &tokens)
{
    stringstream ss(s);
    string temp;
	
	//clear any existing tokens
	tokens.clear();
	
    while (getline(ss, temp, delim)) {
        tokens.push_back(temp);
    }
}

/*
Reads in a csv file containing examples in the form <val1,val2...,val-n,class-label>
Example format for 2d data: 2.334,5.276,-1
The class lable is expected to be some kind of numeric, 1, -1, a probability, or even an integer-class (for multiclasses).
This function doesn't need to know any of that.

Returns: nothing, stores output in vector<double>.
*/
void readCsv(const string& path, vector<vector<double> >& output)
{
	int i;
	fstream dataFile;
	string line;
	vector<string> tokens;
	vector<double> temp;

	dataFile.open(path.c_str(),ios::in);
	if(!dataFile.is_open()){
		cout << "ERROR could not open dataset file: " << path << endl;
	}

	cout << "Building dataset..." << endl;
	//clear any existing data
	output.clear();
	
	while(getline(dataFile, line)){
		tokens.clear();
		tokenize(line,',',tokens);
		
		//build a temp double vector containing the vals from the record, in double form
		temp.clear();
		for(i = 0; i < tokens.size(); i++){
			temp.push_back(std::stod(tokens[i]));
		}
		
		output.push_back(temp);
	}
	cout << "Dataset build complete. Read " << output.size() << " examples." << endl;
}

#ifndef UTIL_HPP
#define UTIL_HPP

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

void tokenize(const string &s, char delim, vector<string> &tokens);
void readCsv(const string& path, vector<vector<double> >& output);

#endif
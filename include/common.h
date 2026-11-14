#ifndef COMMON_H
#define COMMON_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <functional>
#include <limits>

using namespace std;

const vector<string> TARGETS = {
    "Perpetrator Sex", "Perpetrator Age", "Perpetrator Race",
    "Perpetrator Ethnicity", "Weapon"
};

struct Dataset {
    vector<vector<double>> X;
    vector<double> y;
    vector<string> featureNames;
};

vector<string> splitCSV(const string& line);
Dataset readDataset(const string& filename, const string& targetName);
void splitClassification(const vector<double>& y, vector<int>& train, vector<int>& test, unsigned seed = 42);
void splitRegression(int n, vector<int>& train, vector<int>& test, unsigned seed = 42);
void classificationMetrics(const vector<int>& actual, const vector<int>& predicted, int classes,
                           const vector<string>& names = {}, bool perClass = true);
void regressionMetrics(const vector<double>& actual, const vector<double>& predicted);
vector<int> makeClassLabels(const vector<double>& y, vector<double>& originalLabels);

// Text <-> number mappings read from encoding_mappings.json
struct Encoding {
    vector<string> inputColumns;
    map<string, map<string, int>> featureMaps;   // column -> (text -> code)
    map<string, map<string, int>> targetMaps;    // target -> (text -> code)
    map<string, double> medians;                 // numeric column -> median
};
Encoding loadEncoding(const string& path);
string decodeName(const map<string, int>& mapping, int code);   // code -> text (or the number if unknown)
#endif

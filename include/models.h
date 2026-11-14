#ifndef MODELS_H
#define MODELS_H
#include "common.h"

struct BinnedData {
    vector<vector<unsigned char>> bins;
    vector<vector<double>> thresholds;
};

struct TreeNode {
    int feature = -1;
    int splitBin = -1;
    int left = -1;
    int right = -1;
    double value = 0.0;
    vector<double> probs;
};

struct Tree {
    vector<TreeNode> nodes;
};

BinnedData createBins(const vector<vector<double>>& X, const vector<int>& train, int maxBins);
vector<double> predictTreeProb(const Tree& tree, const BinnedData& bins, int row);
double predictTreeValue(const Tree& tree, const BinnedData& bins, int row);

struct RFParams {
    int trees = 40;
    int depth = 20;
    int minSplit = 5;
    int minLeaf = 2;
    int bins = 32;
    int features = 8;
    double bootstrap = 1.0;
};

RFParams paramsFor(const string& target);

class RandomForest {
public:
    bool isRegression = false;
    int classCount = 0;
    RFParams params;
    BinnedData binned;
    vector<Tree> trees;
    void fit(const vector<vector<double>>& X, const vector<double>& y, bool regression, int classes, const RFParams& p, unsigned seed = 42);
    vector<double> predictProba(const vector<double>& row) const;
    double predictValue(const vector<double>& row) const;
    void save(const string& path) const;
    bool load(const string& path);
};

struct XGBParams {
    int bins = 32;
    int depth = 4;
    int rounds = 30;
    double learningRate = 0.1;
    double lambda = 1.0;
    double gamma = 0.0;
    double minChildWeight = 1.0;
};

class XGBoostModel {
public:
    bool isRegression = false;
    int classCount = 0;
    double baseValue = 0.0;
    XGBParams params;
    BinnedData binned;
    vector<vector<Tree>> classTrees;
    vector<Tree> regressionTrees;
    void fit(const vector<vector<double>>& X, const vector<double>& y, bool regression, int classes, const XGBParams& p, unsigned seed = 42);
    vector<double> predictProba(const vector<double>& row) const;
    double predictValue(const vector<double>& row) const;
    void save(const string& path) const;
    bool load(const string& path);
};

class DecisionTree {
public:
    bool isRegression = false;
    int classCount = 0;
    BinnedData binned;
    Tree tree;
    int maxDepth = 20;
    int minLeaf = 2;
    void fit(const vector<vector<double>>& X, const vector<double>& y, bool regression, int classes, unsigned seed = 42);
    vector<double> predictProba(const vector<double>& row) const;
    double predictValue(const vector<double>& row) const;
    void save(const string& path) const;
    bool load(const string& path);
};

vector<double> ensembleProbabilities(const vector<double>& rf, const vector<double>& xgb, const vector<double>& dt);
double ensembleRegression(double rf, double xgb, double dt);
#endif

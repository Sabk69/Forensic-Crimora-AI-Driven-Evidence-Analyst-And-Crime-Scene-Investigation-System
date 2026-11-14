#include "models.h"

static mt19937 rng(42);

BinnedData createBins(const vector<vector<double>>& X, const vector<int>& train, int maxBins) {
    BinnedData result;
    if (X.empty()) return result;
    int n = X.size(), features = X[0].size();
    result.bins.resize(features);
    result.thresholds.resize(features);
    for (int f = 0; f < features; f++) {
        vector<double> values;
        for (int r : train) values.push_back(X[r][f]);
        sort(values.begin(), values.end());
        if (values.empty()) continue;
        for (int b = 1; b < maxBins; b++) {
            int pos = min((int)values.size() - 1, (int)((double)b * values.size() / maxBins));
            double cut = values[pos];
            if (result.thresholds[f].empty() || cut > result.thresholds[f].back()) result.thresholds[f].push_back(cut);
        }
        result.bins[f].resize(n);
        for (int i = 0; i < n; i++) {
            int bin = upper_bound(result.thresholds[f].begin(), result.thresholds[f].end(), X[i][f]) - result.thresholds[f].begin();
            result.bins[f][i] = (unsigned char)bin;
        }
    }
    return result;
}

static vector<unsigned char> binRow(const vector<double>& row, const BinnedData& binned) {
    vector<unsigned char> result(row.size());
    for (int f = 0; f < (int)row.size(); f++) {
        int b = upper_bound(binned.thresholds[f].begin(), binned.thresholds[f].end(), row[f]) - binned.thresholds[f].begin();
        result[f] = (unsigned char)b;
    }
    return result;
}

vector<double> predictTreeProb(const Tree& tree, const BinnedData& bins, int row) {
    int current = 0;
    while (tree.nodes[current].feature != -1) {
        const TreeNode& node = tree.nodes[current];
        current = bins.bins[node.feature][row] <= node.splitBin ? node.left : node.right;
    }
    return tree.nodes[current].probs;
}

double predictTreeValue(const Tree& tree, const BinnedData& bins, int row) {
    int current = 0;
    while (tree.nodes[current].feature != -1) {
        const TreeNode& node = tree.nodes[current];
        current = bins.bins[node.feature][row] <= node.splitBin ? node.left : node.right;
    }
    return tree.nodes[current].value;
}

static vector<int> chooseFeatures(int featureCount, int wanted, mt19937& gen) {
    vector<int> features(featureCount);
    for (int i = 0; i < featureCount; i++) features[i] = i;
    shuffle(features.begin(), features.end(), gen);
    if (wanted <= 0) wanted = max(1, (int)sqrt((double)featureCount));
    if (wanted > featureCount) wanted = featureCount;
    features.resize(wanted);
    return features;
}

static int buildRFClass(Tree& tree, const vector<int>& rows, int depth,
                        const vector<int>& y, const vector<double>& weights,
                        int classes, const BinnedData& bins, const RFParams& p,
                        mt19937& gen, bool allFeatures = false) {
    vector<double> sums(classes, 0.0);
    double total = 0;
    for (int r : rows) { sums[y[r]] += weights[y[r]]; total += weights[y[r]]; }
    TreeNode node;
    node.probs.assign(classes, 0.0);
    if (total > 0) for (int c = 0; c < classes; c++) node.probs[c] = sums[c] / total;
    int index = tree.nodes.size(); tree.nodes.push_back(node);
    double parentGini = 1.0;
    if (total > 0) {
        double sq = 0;
        for (int c = 0; c < classes; c++) { double q = sums[c] / total; sq += q * q; }
        parentGini = 1.0 - sq;
    }
    if (depth >= p.depth || (int)rows.size() < p.minSplit || parentGini <= 1e-9) return index;
    int featureCount = bins.bins.size();
    vector<int> features = allFeatures ? vector<int>() : chooseFeatures(featureCount, p.features, gen);
    if (allFeatures) for (int f = 0; f < featureCount; f++) features.push_back(f);
    double bestGain = 0; int bestFeature = -1, bestBin = -1;
    for (int f : features) {
        int binCount = bins.thresholds[f].size() + 1;
        vector<double> hist(binCount * classes, 0.0);
        vector<int> counts(binCount, 0);
        for (int r : rows) { int b = bins.bins[f][r]; hist[b * classes + y[r]] += weights[y[r]]; counts[b]++; }
        vector<double> left(classes, 0.0); double leftWeight = 0; int leftCount = 0;
        for (int b = 0; b < binCount - 1; b++) {
            leftCount += counts[b];
            for (int c = 0; c < classes; c++) { left[c] += hist[b * classes + c]; leftWeight += hist[b * classes + c]; }
            int rightCount = rows.size() - leftCount; double rightWeight = total - leftWeight;
            if (leftCount < p.minLeaf || rightCount < p.minLeaf || leftWeight <= 0 || rightWeight <= 0) continue;
            double lsum = 0, rsum = 0;
            for (int c = 0; c < classes; c++) {
                double lp = left[c] / leftWeight, rp = (sums[c] - left[c]) / rightWeight;
                lsum += lp * lp; rsum += rp * rp;
            }
            double gain = parentGini - ((leftWeight / total) * (1 - lsum) + (rightWeight / total) * (1 - rsum));
            if (gain > bestGain) { bestGain = gain; bestFeature = f; bestBin = b; }
        }
    }
    if (bestFeature < 0) return index;
    vector<int> leftRows, rightRows;
    for (int r : rows) {
        if (bins.bins[bestFeature][r] <= bestBin) leftRows.push_back(r);
        else rightRows.push_back(r);
    }
    if ((int)leftRows.size() < p.minLeaf || (int)rightRows.size() < p.minLeaf) return index;
    tree.nodes[index].feature = bestFeature; tree.nodes[index].splitBin = bestBin;
    int l = buildRFClass(tree, leftRows, depth + 1, y, weights, classes, bins, p, gen, allFeatures);
    int r = buildRFClass(tree, rightRows, depth + 1, y, weights, classes, bins, p, gen, allFeatures);
    tree.nodes[index].left = l; tree.nodes[index].right = r;
    return index;
}

static int buildRFReg(Tree& tree, const vector<int>& rows, int depth, const vector<double>& y,
                      const BinnedData& bins, const RFParams& p, mt19937& gen, bool allFeatures = false) {
    double sum = 0; for (int r : rows) sum += y[r];
    int n = rows.size();
    TreeNode node; node.value = n ? sum / n : 0;
    int index = tree.nodes.size(); tree.nodes.push_back(node);
    if (depth >= p.depth || n < p.minSplit) return index;
    int featureCount = bins.bins.size();
    vector<int> features = allFeatures ? vector<int>() : chooseFeatures(featureCount, p.features, gen);
    if (allFeatures) for (int f = 0; f < featureCount; f++) features.push_back(f);
    double base = sum * sum / n, bestGain = 1e-9; int bestFeature = -1, bestBin = -1;
    for (int f : features) {
        int countBins = bins.thresholds[f].size() + 1;
        vector<double> sums(countBins, 0); vector<int> counts(countBins, 0);
        for (int r : rows) { int b = bins.bins[f][r]; sums[b] += y[r]; counts[b]++; }
        double leftSum = 0; int leftCount = 0;
        for (int b = 0; b < countBins - 1; b++) {
            leftSum += sums[b]; leftCount += counts[b];
            int rightCount = n - leftCount; double rightSum = sum - leftSum;
            if (leftCount < p.minLeaf || rightCount < p.minLeaf) continue;
            double gain = leftSum * leftSum / leftCount + rightSum * rightSum / rightCount - base;
            if (gain > bestGain) { bestGain = gain; bestFeature = f; bestBin = b; }
        }
    }
    if (bestFeature < 0) return index;
    vector<int> leftRows, rightRows;
    for (int r : rows) { if (bins.bins[bestFeature][r] <= bestBin) leftRows.push_back(r); else rightRows.push_back(r); }
    if ((int)leftRows.size() < p.minLeaf || (int)rightRows.size() < p.minLeaf) return index;
    tree.nodes[index].feature = bestFeature; tree.nodes[index].splitBin = bestBin;
    int l = buildRFReg(tree, leftRows, depth + 1, y, bins, p, gen, allFeatures);
    int r = buildRFReg(tree, rightRows, depth + 1, y, bins, p, gen, allFeatures);
    tree.nodes[index].left = l; tree.nodes[index].right = r;
    return index;
}

RFParams paramsFor(const string& target) {
    if (target == "Perpetrator Sex") return {40,20,5,2,32,8,1.0};
    if (target == "Perpetrator Age") return {30,14,5,30,32,6,1.0};
    if (target == "Perpetrator Race") return {40,20,5,2,32,8,1.0};
    if (target == "Perpetrator Ethnicity") return {40,10,5,2,32,3,1.0};
    return {40,20,5,2,32,8,1.0};
}

void RandomForest::fit(const vector<vector<double>>& X, const vector<double>& target, bool regression, int classes, const RFParams& p, unsigned seed) {
    isRegression = regression; classCount = classes; params = p; trees.clear();
    vector<int> all(X.size()); for (int i = 0; i < (int)X.size(); i++) all[i] = i;
    binned = createBins(X, all, params.bins);
    mt19937 gen(seed);
    vector<int> yClass;
    vector<double> weights;
    if (!regression) {
        yClass.resize(target.size()); vector<int> counts(classes, 0);
        for (int i = 0; i < (int)target.size(); i++) { yClass[i] = (int)target[i]; counts[yClass[i]]++; }
        weights.resize(classes, 1.0);
        for (int c = 0; c < classes; c++) if (counts[c]) weights[c] = (double)target.size() / (classes * counts[c]);
    }
    int sampleSize = max(1, (int)(X.size() * params.bootstrap));
    for (int t = 0; t < params.trees; t++) {
        vector<int> sample(sampleSize);
        uniform_int_distribution<int> pick(0, (int)X.size() - 1);
        for (int i = 0; i < sampleSize; i++) sample[i] = pick(gen);
        Tree tree;
        if (regression) buildRFReg(tree, sample, 0, target, binned, params, gen);
        else buildRFClass(tree, sample, 0, yClass, weights, classes, binned, params, gen);
        trees.push_back(tree);
        if ((t + 1) % 10 == 0) cout << "  RF trees: " << t + 1 << "/" << params.trees << "\n";
    }
}

vector<double> RandomForest::predictProba(const vector<double>& row) const {
    vector<double> result(classCount, 0.0);
    if (trees.empty()) return result;
    vector<unsigned char> binsRow = binRow(row, binned);
    for (const Tree& tree : trees) {
        int current = 0;
        while (tree.nodes[current].feature != -1) {
            const TreeNode& node = tree.nodes[current];
            current = binsRow[node.feature] <= node.splitBin ? node.left : node.right;
        }
        for (int c = 0; c < classCount; c++) result[c] += tree.nodes[current].probs[c];
    }
    for (double& v : result) v /= trees.size();
    return result;
}

double RandomForest::predictValue(const vector<double>& row) const {
    if (trees.empty()) return 0;
    vector<unsigned char> binsRow = binRow(row, binned);
    double sum = 0;
    for (const Tree& tree : trees) {
        int current = 0;
        while (tree.nodes[current].feature != -1) { const TreeNode& n = tree.nodes[current]; current = binsRow[n.feature] <= n.splitBin ? n.left : n.right; }
        sum += tree.nodes[current].value;
    }
    return sum / trees.size();
}

// XGBoost-style histogram tree. It uses the same G/H gain formula as the supplied old code.
static int buildXGBNode(Tree& tree, const vector<int>& rows, int depth,
                        const vector<double>& G, const vector<double>& H,
                        const BinnedData& bins, const XGBParams& p) {
    double totalG = 0, totalH = 0;
    for (int r : rows) { totalG += G[r]; totalH += H[r]; }
    TreeNode node; node.value = -totalG / (totalH + p.lambda);
    int index = tree.nodes.size(); tree.nodes.push_back(node);
    if (depth >= p.depth || rows.size() < 2) return index;
    int bestFeature = -1, bestBin = -1; double bestGain = 0;
    for (int f = 0; f < (int)bins.bins.size(); f++) {
        int countBins = bins.thresholds[f].size() + 1;
        vector<double> histG(countBins, 0), histH(countBins, 0);
        for (int r : rows) { int b = bins.bins[f][r]; histG[b] += G[r]; histH[b] += H[r]; }
        double leftG = 0, leftH = 0;
        for (int b = 0; b < countBins - 1; b++) {
            leftG += histG[b]; leftH += histH[b];
            double rightG = totalG - leftG, rightH = totalH - leftH;
            if (leftH < p.minChildWeight || rightH < p.minChildWeight) continue;
            double gain = 0.5 * (leftG * leftG / (leftH + p.lambda) + rightG * rightG / (rightH + p.lambda) - totalG * totalG / (totalH + p.lambda)) - p.gamma;
            if (gain > bestGain) { bestGain = gain; bestFeature = f; bestBin = b; }
        }
    }
    if (bestFeature < 0) return index;
    vector<int> leftRows, rightRows;
    for (int r : rows) { if (bins.bins[bestFeature][r] <= bestBin) leftRows.push_back(r); else rightRows.push_back(r); }
    if (leftRows.empty() || rightRows.empty()) return index;
    tree.nodes[index].feature = bestFeature; tree.nodes[index].splitBin = bestBin;
    int l = buildXGBNode(tree, leftRows, depth + 1, G, H, bins, p);
    int r = buildXGBNode(tree, rightRows, depth + 1, G, H, bins, p);
    tree.nodes[index].left = l; tree.nodes[index].right = r;
    return index;
}

static vector<double> softmaxValues(const vector<double>& logits) {
    vector<double> p(logits.size());
    if (logits.empty()) return p;
    double mx = *max_element(logits.begin(), logits.end()), total = 0;
    for (int i = 0; i < (int)logits.size(); i++) { p[i] = exp(logits[i] - mx); total += p[i]; }
    if (total > 0) for (double& v : p) v /= total;
    return p;
}

void XGBoostModel::fit(const vector<vector<double>>& X, const vector<double>& target, bool regression, int classes, const XGBParams& p, unsigned seed) {
    (void)seed;
    isRegression = regression; classCount = classes; params = p;
    classTrees.clear(); regressionTrees.clear();
    vector<int> all(X.size()); for (int i = 0; i < (int)X.size(); i++) all[i] = i;
    binned = createBins(X, all, params.bins);
    if (regression) {
        baseValue = 0; for (double v : target) baseValue += v; baseValue /= target.size();
        vector<double> pred(target.size(), baseValue);
        for (int round = 0; round < params.rounds; round++) {
            vector<double> G(target.size()), H(target.size(), 1.0);
            for (int i = 0; i < (int)target.size(); i++) G[i] = pred[i] - target[i];
            Tree tree; buildXGBNode(tree, all, 0, G, H, binned, params);
            for (int i = 0; i < (int)target.size(); i++) pred[i] += params.learningRate * predictTreeValue(tree, binned, i);
            regressionTrees.push_back(tree);
            if ((round + 1) % 5 == 0) cout << "  XGB rounds: " << round + 1 << "/" << params.rounds << "\n";
        }
        return;
    }
    vector<int> y(target.size()), counts(classes, 0);
    for (int i = 0; i < (int)target.size(); i++) { y[i] = (int)target[i]; counts[y[i]]++; }
    vector<double> weights(classes, 1.0);
    for (int c = 0; c < classes; c++) if (counts[c]) weights[c] = (double)target.size() / (classes * counts[c]);
    vector<vector<double>> pred(target.size(), vector<double>(classes, 0.0));
    classTrees.resize(classes);
    for (int round = 0; round < params.rounds; round++) {
        for (int c = 0; c < classes; c++) {
            vector<double> G(target.size(), 0), H(target.size(), 0);
            // Matches the supplied implementation: each class is updated in sequence within a round.
            for (int i = 0; i < (int)target.size(); i++) {
                vector<double> probability = softmaxValues(pred[i]);
                double q = probability[c], w = weights[y[i]];
                G[i] = w * (q - (y[i] == c ? 1.0 : 0.0));
                H[i] = w * q * (1.0 - q);
            }
            Tree tree; buildXGBNode(tree, all, 0, G, H, binned, params);
            for (int i = 0; i < (int)target.size(); i++) pred[i][c] += params.learningRate * predictTreeValue(tree, binned, i);
            classTrees[c].push_back(tree);
        }
        if ((round + 1) % 5 == 0) cout << "  XGB rounds: " << round + 1 << "/" << params.rounds << "\n";
    }
}

vector<double> XGBoostModel::predictProba(const vector<double>& row) const {
    vector<unsigned char> binsRow = binRow(row, binned);
    vector<double> logits(classCount, 0.0);
    for (int c = 0; c < classCount; c++) {
        for (const Tree& tree : classTrees[c]) {
            int current = 0;
            while (tree.nodes[current].feature != -1) { const TreeNode& n = tree.nodes[current]; current = binsRow[n.feature] <= n.splitBin ? n.left : n.right; }
            logits[c] += params.learningRate * tree.nodes[current].value;
        }
    }
    return softmaxValues(logits);
}

double XGBoostModel::predictValue(const vector<double>& row) const {
    vector<unsigned char> binsRow = binRow(row, binned);
    double result = baseValue;
    for (const Tree& tree : regressionTrees) {
        int current = 0;
        while (tree.nodes[current].feature != -1) { const TreeNode& n = tree.nodes[current]; current = binsRow[n.feature] <= n.splitBin ? n.left : n.right; }
        result += params.learningRate * tree.nodes[current].value;
    }
    return result;
}

void DecisionTree::fit(const vector<vector<double>>& X, const vector<double>& target, bool regression, int classes, unsigned seed) {
    isRegression = regression; classCount = classes;
    vector<int> all(X.size()); for (int i = 0; i < (int)X.size(); i++) all[i] = i;
    binned = createBins(X, all, 32);
    RFParams p; p.trees = 1; p.depth = maxDepth; p.minSplit = 5; p.minLeaf = minLeaf; p.bins = 32; p.features = (int)X[0].size();
    mt19937 gen(seed);
    if (regression) buildRFReg(tree, all, 0, target, binned, p, gen, true);
    else {
        vector<int> y(target.size()), counts(classes, 0);
        for (int i = 0; i < (int)target.size(); i++) { y[i] = (int)target[i]; counts[y[i]]++; }
        vector<double> weights(classes, 1.0);
        for (int c = 0; c < classes; c++) if (counts[c]) weights[c] = (double)target.size() / (classes * counts[c]);
        buildRFClass(tree, all, 0, y, weights, classes, binned, p, gen, true);
    }
}

vector<double> DecisionTree::predictProba(const vector<double>& row) const {
    vector<unsigned char> binsRow = binRow(row, binned); int current = 0;
    while (tree.nodes[current].feature != -1) { const TreeNode& n = tree.nodes[current]; current = binsRow[n.feature] <= n.splitBin ? n.left : n.right; }
    return tree.nodes[current].probs;
}

double DecisionTree::predictValue(const vector<double>& row) const {
    vector<unsigned char> binsRow = binRow(row, binned); int current = 0;
    while (tree.nodes[current].feature != -1) { const TreeNode& n = tree.nodes[current]; current = binsRow[n.feature] <= n.splitBin ? n.left : n.right; }
    return tree.nodes[current].value;
}

vector<double> ensembleProbabilities(const vector<double>& rf, const vector<double>& xgb, const vector<double>& dt) {
    vector<double> result(rf.size(), 0.0);
    for (int i = 0; i < (int)result.size(); i++) result[i] = (rf[i] + xgb[i] + dt[i]) / 3.0;
    return result;
}
double ensembleRegression(double rf, double xgb, double dt) { return (rf + xgb + dt) / 3.0; }

// Shared text serialization helpers.
static void saveBins(ofstream& out, const BinnedData& b) {
    out << b.thresholds.size() << "\n";
    for (const auto& v : b.thresholds) { out << v.size(); for (double x : v) out << " " << setprecision(17) << x; out << "\n"; }
}
static bool loadBins(ifstream& in, BinnedData& b) {
    int features; if (!(in >> features)) return false;
    b.thresholds.resize(features); b.bins.resize(features);
    for (int f = 0; f < features; f++) { int n; in >> n; b.thresholds[f].resize(n); for (double& x : b.thresholds[f]) in >> x; }
    return (bool)in;
}
static void saveTree(ofstream& out, const Tree& t) {
    out << t.nodes.size() << "\n";
    for (const TreeNode& n : t.nodes) {
        out << n.feature << " " << n.splitBin << " " << n.left << " " << n.right << " " << setprecision(17) << n.value << " " << n.probs.size();
        for (double p : n.probs) out << " " << setprecision(17) << p;
        out << "\n";
    }
}
static bool loadTree(ifstream& in, Tree& t) {
    int n; if (!(in >> n)) return false; t.nodes.resize(n);
    for (TreeNode& node : t.nodes) { int count; in >> node.feature >> node.splitBin >> node.left >> node.right >> node.value >> count; node.probs.resize(count); for (double& p : node.probs) in >> p; }
    return (bool)in;
}

void RandomForest::save(const string& path) const {
    ofstream out(path); if (!out) throw runtime_error("Cannot save model: " + path);
    out << "HOMICIDE_RF_1\n" << isRegression << " " << classCount << "\n";
    out << params.trees << " " << params.depth << " " << params.minSplit << " " << params.minLeaf << " " << params.bins << " " << params.features << " " << params.bootstrap << "\n";
    saveBins(out, binned); out << trees.size() << "\n"; for (const Tree& t : trees) saveTree(out, t);
}
bool RandomForest::load(const string& path) {
    ifstream in(path); string magic; if (!(in >> magic) || magic != "HOMICIDE_RF_1") return false;
    in >> isRegression >> classCount >> params.trees >> params.depth >> params.minSplit >> params.minLeaf >> params.bins >> params.features >> params.bootstrap;
    if (!loadBins(in, binned)) return false;
    int n; in >> n;
    trees.resize(n);
    for (Tree& t : trees) if (!loadTree(in, t)) return false;
    return true;
}
void XGBoostModel::save(const string& path) const {
    ofstream out(path); if (!out) throw runtime_error("Cannot save model: " + path);
    out << "HOMICIDE_XGB_1\n" << isRegression << " " << classCount << " " << setprecision(17) << baseValue << "\n";
    out << params.bins << " " << params.depth << " " << params.rounds << " " << params.learningRate << " " << params.lambda << " " << params.gamma << " " << params.minChildWeight << "\n";
    saveBins(out, binned);
    out << classTrees.size() << "\n"; for (const auto& group : classTrees) { out << group.size() << "\n"; for (const Tree& t : group) saveTree(out, t); }
    out << regressionTrees.size() << "\n"; for (const Tree& t : regressionTrees) saveTree(out, t);
}
bool XGBoostModel::load(const string& path) {
    ifstream in(path); string magic; if (!(in >> magic) || magic != "HOMICIDE_XGB_1") return false;
    in >> isRegression >> classCount >> baseValue >> params.bins >> params.depth >> params.rounds >> params.learningRate >> params.lambda >> params.gamma >> params.minChildWeight;
    if (!loadBins(in, binned)) return false;
    int groups; in >> groups;
    classTrees.resize(groups);
    for (auto& group : classTrees) { int n; in >> n; group.resize(n); for (Tree& t : group) if (!loadTree(in, t)) return false; }
    int n; in >> n; regressionTrees.resize(n); for (Tree& t : regressionTrees) if (!loadTree(in, t)) return false; return true;
}
void DecisionTree::save(const string& path) const {
    ofstream out(path); if (!out) throw runtime_error("Cannot save model: " + path);
    out << "HOMICIDE_DT_1\n" << isRegression << " " << classCount << " " << maxDepth << " " << minLeaf << "\n"; saveBins(out, binned); saveTree(out, tree);
}
bool DecisionTree::load(const string& path) {
    ifstream in(path); string magic; if (!(in >> magic) || magic != "HOMICIDE_DT_1") return false;
    in >> isRegression >> classCount >> maxDepth >> minLeaf; if (!loadBins(in, binned)) return false; return loadTree(in, tree);
}

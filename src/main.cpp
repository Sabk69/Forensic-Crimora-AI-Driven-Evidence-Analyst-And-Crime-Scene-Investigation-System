#include "common.h"
#include "models.h"
#include <direct.h>
#include<chrono>
static vector<vector<double>> takeRows(const vector<vector<double>>& X, const vector<int>& ids) {
    vector<vector<double>> result; result.reserve(ids.size());
    for (int i : ids) result.push_back(X[i]);
    return result;
}
static vector<double> takeValues(const vector<double>& y, const vector<int>& ids) {
    vector<double> result; result.reserve(ids.size());
    for (int i : ids) result.push_back(y[i]);
    return result;
}
static vector<int> takeIntValues(const vector<int>& y, const vector<int>& ids) {
    vector<int> result; result.reserve(ids.size());
    for (int i : ids) result.push_back(y[i]);
    return result;
}
static string safeName(const string& target) {
    string s = target;
    for (char& c : s) if (c == ' ') c = '_';
    return s;
}
static string jsonEscape(const string& s) {
    string out;
    for (char c : s) {
        if (c == '\\' || c == '"') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    return out;
}
static void saveSchema(const string& folder, const string& target, const vector<string>& features, const vector<double>& labels) {
    ofstream out(folder + "/" + safeName(target) + "_schema.txt");
    out << features.size() << "\n";
    for (const string& f : features) out << quoted(f) << "\n";
    out << labels.size() << "\n";
    for (double v : labels) out << setprecision(17) << v << "\n";
}
static double actualSplitThreshold(const BinnedData& bins, const TreeNode& node) {
    if (node.feature < 0 || node.feature >= (int)bins.thresholds.size()) return 0.0;
    if (node.splitBin < 0 || node.splitBin >= (int)bins.thresholds[node.feature].size()) return 0.0;
    return bins.thresholds[node.feature][node.splitBin];
}
static void writeBinThresholdsText(ofstream& out, const BinnedData& bins, const vector<string>& features, const string& title) {
    out << "\n" << title << " BIN CUT POINTS\n";
    for (int f = 0; f < (int)bins.thresholds.size(); f++) {
        out << "Feature " << f << " (" << features[f] << "): ";
        for (double value : bins.thresholds[f]) out << setprecision(17) << value << " ";
        out << "\n";
    }
}
static void writeTreeText(ofstream& out, const Tree& tree, const BinnedData& bins, const string& title) {
    out << "\n" << title << "\n";
    for (int i = 0; i < (int)tree.nodes.size(); i++) {
        const TreeNode& n = tree.nodes[i];
        out << "Node " << i << ": ";
        if (n.feature == -1) {
            out << "LEAF value=" << setprecision(17) << n.value;
            if (!n.probs.empty()) { out << " probabilities="; for (double p : n.probs) out << p << " "; }
        } else {
            out << "IF feature_index=" << n.feature << " bin <= " << n.splitBin
                << " (actual_value <= " << setprecision(17) << actualSplitThreshold(bins, n) << ") THEN node " << n.left << " ELSE node " << n.right;
        }
        out << "\n";
    }
}
static void writeTreeJson(ofstream& out, const Tree& tree, const BinnedData& bins) {
    out << "[";
    for (int i = 0; i < (int)tree.nodes.size(); i++) {
        if (i) out << ",";
        const TreeNode& n = tree.nodes[i];
        out << "{\"feature\":" << n.feature << ",\"split_bin\":" << n.splitBin
            << ",\"left\":" << n.left << ",\"right\":" << n.right
            << ",\"value\":" << setprecision(17) << n.value
            << ",\"split_threshold\":" << (n.feature == -1 ? 0.0 : actualSplitThreshold(bins, n)) << ",\"probabilities\":[";
        for (int j = 0; j < (int)n.probs.size(); j++) { if (j) out << ","; out << setprecision(17) << n.probs[j]; }
        out << "]}";
    }
    out << "]";
}
static void savePatterns(const string& target, const vector<string>& features,
                         const RandomForest& rf, const XGBoostModel& xgb, const DecisionTree& dt) {
    string prefix = "models/" + safeName(target);
    ofstream txt(prefix + "_patterns.txt");
    ofstream js(prefix + "_patterns.json");
    if (!txt || !js) throw runtime_error("Could not write pattern files for " + target);
    txt << "LEARNED MODEL PATTERNS: " << target << "\nFeature indexes:\n";
    for (int i = 0; i < (int)features.size(); i++) txt << i << " = " << features[i] << "\n";
    txt << "\nRANDOM FOREST: " << rf.trees.size() << " trees\n";
    writeBinThresholdsText(txt, rf.binned, features, "RANDOM FOREST");
    for (int i = 0; i < (int)rf.trees.size(); i++) writeTreeText(txt, rf.trees[i], rf.binned, "RF TREE " + to_string(i));
    txt << "\nXGBOOST CLASS TREES\n";
    writeBinThresholdsText(txt, xgb.binned, features, "XGBOOST");
    for (int c = 0; c < (int)xgb.classTrees.size(); c++)
        for (int r = 0; r < (int)xgb.classTrees[c].size(); r++)
            writeTreeText(txt, xgb.classTrees[c][r], xgb.binned, "XGB CLASS " + to_string(c) + " ROUND " + to_string(r));
    for (int r = 0; r < (int)xgb.regressionTrees.size(); r++)
        writeTreeText(txt, xgb.regressionTrees[r], xgb.binned, "XGB REGRESSION ROUND " + to_string(r));
    writeBinThresholdsText(txt, dt.binned, features, "DECISION TREE");
    writeTreeText(txt, dt.tree, dt.binned, "DECISION TREE");

    js << "{\n\"target\":\"" << jsonEscape(target) << "\",\n\"features\":[";
    for (int i = 0; i < (int)features.size(); i++) { if (i) js << ","; js << "\"" << jsonEscape(features[i]) << "\""; }
    js << "],\n\"random_forest_bin_thresholds\":[";
    for (int f = 0; f < (int)rf.binned.thresholds.size(); f++) { if (f) js << ","; js << "["; for (int j = 0; j < (int)rf.binned.thresholds[f].size(); j++) { if (j) js << ","; js << setprecision(17) << rf.binned.thresholds[f][j]; } js << "]"; }
    js << "],\n\"random_forest\":[";
    for (int i = 0; i < (int)rf.trees.size(); i++) { if (i) js << ","; writeTreeJson(js, rf.trees[i], rf.binned); }
    js << "],\n\"xgboost_bin_thresholds\":[";
    for (int f = 0; f < (int)xgb.binned.thresholds.size(); f++) { if (f) js << ","; js << "["; for (int j = 0; j < (int)xgb.binned.thresholds[f].size(); j++) { if (j) js << ","; js << setprecision(17) << xgb.binned.thresholds[f][j]; } js << "]"; }
    js << "],\n\"xgboost_class_trees\":[";
    for (int c = 0; c < (int)xgb.classTrees.size(); c++) { if (c) js << ","; js << "[";
        for (int r = 0; r < (int)xgb.classTrees[c].size(); r++) { if (r) js << ","; writeTreeJson(js, xgb.classTrees[c][r], xgb.binned); }
        js << "]";
    }
    js << "],\n\"xgboost_regression_trees\":[";
    for (int r = 0; r < (int)xgb.regressionTrees.size(); r++) { if (r) js << ","; writeTreeJson(js, xgb.regressionTrees[r], xgb.binned); }
    js << "],\n\"decision_tree_bin_thresholds\":[";
    for (int f = 0; f < (int)dt.binned.thresholds.size(); f++) { if (f) js << ","; js << "["; for (int j = 0; j < (int)dt.binned.thresholds[f].size(); j++) { if (j) js << ","; js << setprecision(17) << dt.binned.thresholds[f][j]; } js << "]"; }
    js << "],\n\"decision_tree\":"; writeTreeJson(js, dt.tree, dt.binned);
    js << "\n}\n";
}

static Encoding enc;           // text <-> number mappings from encoding_mappings.json
static bool encLoaded = false;

static vector<int> predictClasses(const vector<vector<double>>& X, const function<vector<double>(const vector<double>&)>& proba) {
    vector<int> pred; pred.reserve(X.size());
    for (const auto& row : X) { vector<double> p = proba(row); pred.push_back(max_element(p.begin(), p.end()) - p.begin()); }
    return pred;
}
static vector<double> predictValues(const vector<vector<double>>& X, const function<double(const vector<double>&)>& value) {
    vector<double> pred; pred.reserve(X.size());
    for (const auto& row : X) pred.push_back(value(row));
    return pred;
}

static void trainTarget(const string& filename, const string& target, bool saveModels) {
    cout << "\n==================================================\nTARGET: " << target << "\n==================================================\n";
    Dataset data = readDataset(filename, target);
    if (data.X.empty()) { cout << "No usable rows. Skipping.\n"; return; }
    cout << "Usable rows: " << data.y.size() << "\nFeatures: " << data.featureNames.size() << "\n";
    bool regression = target == "Perpetrator Age";
    vector<int> train, test;
    if (regression) splitRegression(data.y.size(), train, test, 42);
    else splitClassification(data.y, train, test, 42);
    cout << "Training rows: " << train.size() << "\nTesting rows: " << test.size() << "\n";

    vector<vector<double>> Xtr = takeRows(data.X, train);
    vector<vector<double>> Xte = takeRows(data.X, test);
    vector<double> ytrOriginal = takeValues(data.y, train);
    vector<double> yteOriginal = takeValues(data.y, test);
    vector<double> originalLabels;
    vector<double> ytr;
    vector<int> yteClass;
    if (regression) ytr = ytrOriginal;
    else {
        vector<int> allLabels = makeClassLabels(data.y, originalLabels);
        vector<int> trainLabels = takeIntValues(allLabels, train);
        ytr.reserve(trainLabels.size());
        for (int v : trainLabels) ytr.push_back(v);
        yteClass = takeIntValues(allLabels, test);
        cout << "Classes: " << originalLabels.size() << "\n";
    }
    int classes = regression ? 0 : (int)originalLabels.size();

    RandomForest rf; RFParams rp = paramsFor(target);
    cout << "\nTraining Random Forest...\n"; rf.fit(Xtr, ytr, regression, classes, rp, 42);
    XGBoostModel xgb; XGBParams xp;
    cout << "\nTraining XGBoost...\n"; xgb.fit(Xtr, ytr, regression, classes, xp, 42);
    DecisionTree dt;
    cout << "\nTraining Decision Tree...\n"; dt.fit(Xtr, ytr, regression, classes, 42);

    // readable class names for the per-class table (falls back to "Class N" if the JSON is missing)
    vector<string> classNames;
    if (encLoaded && enc.targetMaps.count(target))
        for (double code : originalLabels) classNames.push_back(decodeName(enc.targetMaps[target], (int)code));
    vector<int> trainClass;
    for (double v : ytr) trainClass.push_back((int)v);

    typedef function<vector<double>(const vector<double>&)> ProbaFn;
    typedef function<double(const vector<double>&)> ValueFn;
    struct Entry { string title; ProbaFn proba; ValueFn value; };
    vector<Entry> models = {
        {"Random Forest", [&](const vector<double>& r) { return rf.predictProba(r); }, [&](const vector<double>& r) { return rf.predictValue(r); }},
        {"XGBoost", [&](const vector<double>& r) { return xgb.predictProba(r); }, [&](const vector<double>& r) { return xgb.predictValue(r); }},
        {"Decision Tree", [&](const vector<double>& r) { return dt.predictProba(r); }, [&](const vector<double>& r) { return dt.predictValue(r); }},
        {"Three-model ensemble",
         [&](const vector<double>& r) { return ensembleProbabilities(rf.predictProba(r), xgb.predictProba(r), dt.predictProba(r)); },
         [&](const vector<double>& r) { return ensembleRegression(rf.predictValue(r), xgb.predictValue(r), dt.predictValue(r)); }}};

    for (const Entry& m : models) {
        cout << "\n=== " << m.title << " ===\n";
        cout << "\n[TRAINING SCORES]\n";
        if (regression) regressionMetrics(ytrOriginal, predictValues(Xtr, m.value));
        else classificationMetrics(trainClass, predictClasses(Xtr, m.proba), classes, classNames, false);
        cout << "\n[TEST SCORES]\n";
        if (regression) regressionMetrics(yteOriginal, predictValues(Xte, m.value));
        else classificationMetrics(yteClass, predictClasses(Xte, m.proba), classes, classNames, true);
    }

    if (saveModels) {
        string prefix = "models/" + safeName(target);
        rf.save(prefix + "_rf.model"); xgb.save(prefix + "_xgb.model"); dt.save(prefix + "_dt.model");
        saveSchema("models", target, data.featureNames, originalLabels);
        savePatterns(target, data.featureNames, rf, xgb, dt);
        cout << "\nSaved learned patterns and model files for " << target << ":\n"
             << "  " << prefix << "_patterns.txt\n  " << prefix << "_patterns.json\n";
    } else cout << "\nModels/patterns were not saved. They will be lost when the program closes.\n";
}

static vector<string> readSchemaNames(const string& path, vector<double>& labels) {
    ifstream in(path); if (!in) throw runtime_error("Missing saved schema: " + path + ". Train and save models first.");
    int n; in >> n; vector<string> names(n);
    for (string& s : names) in >> quoted(s);
    int count; in >> count; labels.resize(count); for (double& v : labels) in >> v;
    return names;
}
static vector<vector<double>> readFeatureRows(const string& path, const vector<string>& names) {
    ifstream in(path); if (!in) throw runtime_error("Could not open CSV: " + path);
    string line; if (!getline(in, line)) throw runtime_error("CSV is empty: " + path);
    vector<string> header = splitCSV(line); map<string,int> pos;
    for (int i = 0; i < (int)header.size(); i++) pos[header[i]] = i;
    for (const string& name : names) if (!pos.count(name)) throw runtime_error("Missing feature column in CSV: " + name);
    vector<vector<double>> rows;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> values = splitCSV(line); if (values.size() != header.size()) continue;
        vector<double> row;
        for (const string& name : names) {
            double value = 0;
            try { size_t used = 0; value = stod(values[pos[name]], &used); if (used != values[pos[name]].size()) value = 0; }
            catch (...) { value = 0; }
            row.push_back(value);
        }
        rows.push_back(row);
    }
    return rows;
}
static string lower(string s) { for (char& c : s) c = tolower((unsigned char)c); return s; }
static string trim(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == string::npos ? "" : s.substr(a, b - a + 1);
}
static string numberText(double v) {                    // 1995.0000 -> 1995
    ostringstream o; o << (v == (long long)v ? fixed : defaultfloat) << setprecision(v == (long long)v ? 0 : 2) << v; return o.str();
}
static string readLine(const string& prompt) {
    cout << prompt;
    string line;
    if (!getline(cin, line)) throw runtime_error("Input ended.");
    return trim(line);
}

// prints one sample in plain text (codes are turned back into names)
static void printSample(ostream& out, const vector<double>& row, const vector<string>& features) {
    out << "Sample details:\n";
    for (int f = 0; f < (int)features.size(); f++) {
        string value = numberText(row[f]);
        auto it = enc.featureMaps.find(features[f]);
        if (it != enc.featureMaps.end()) value = decodeName(it->second, (int)row[f]);
        out << "  " << left << setw(18) << features[f] << ": " << value << "\n";
    }
}

static void printOptions(const map<string, int>& options) {
    bool first = true;
    for (const auto& item : options) { cout << (first ? "" : ", ") << item.first; first = false; }
    cout << "\n";
}
// allowed range for each numeric column (min, max) - taken from the dataset
static const map<string, pair<int,int>> NUMBER_RANGES = {
    {"Year", {1980, 2014}}, {"Month", {1, 12}},
    {"Victim Age", {0, 99}}, {"Victim Count", {1, 11}}
};
// manual entry: type text (Male, Sheriff, 1995...), r = random value for that column, ? = show choices
static vector<double> enterSample(const vector<string>& features, const vector<vector<double>>& rows, mt19937& rng) {
    cout << "\nType each value in plain text (for example: Male, Sheriff, Texas, 1995).\n"
         << "  r = pick a random value for that column     ? = show the allowed choices\n\n";
    vector<double> row;
    for (int f = 0; f < (int)features.size(); f++) {
        const string& name = features[f];
        auto cat = enc.featureMaps.find(name);
        if (cat != enc.featureMaps.end() && cat->second.size() <= 10) { cout << "(" << name << " choices: "; printOptions(cat->second); cout << ")\n"; }
                auto range = NUMBER_RANGES.find(name);
        if (range != NUMBER_RANGES.end()) cout << "(" << name << " must be a whole number from " << range->second.first << " to " << range->second.second << ")\n";
        while (true) {
            string text = readLine(name + ": ");
            if (lower(text) == "r") {
                if (cat != enc.featureMaps.end()) {
                    auto pick = cat->second.begin();
                    advance(pick, rng() % cat->second.size());
                    row.push_back(pick->second);
                    cout << "   -> random " << name << ": " << pick->first << "\n";
                } else {
                    double value = rows.empty() ? enc.medians[name] : rows[rng() % rows.size()][f];   // a real value from the dataset
                    row.push_back(value);
                    cout << "   -> random " << name << ": " << numberText(value) << "\n";
                }
                break;
            }
            if (text == "?") {
                if (cat != enc.featureMaps.end()) { cout << "Choices: "; printOptions(cat->second); }
                else cout << "Type a number.\n";
                continue;
            }
            if (cat != enc.featureMaps.end()) {
                bool found = false;
                for (const auto& item : cat->second)
                    if (lower(item.first) == lower(text)) { row.push_back(item.second); found = true; break; }
                if (found) break;
                cout << "   '" << text << "' is not a known " << name << ". Type ? to see the choices, or r for random.\n";
            } else {
                try {
                                       size_t used = 0; double value = stod(text, &used);
                    if (used != text.size()) throw invalid_argument("extra text");
                    if (range != NUMBER_RANGES.end() &&
                        (value < range->second.first || value > range->second.second || value != floor(value))) {
                        cout << "   " << name << " must be a whole number from " << range->second.first
                             << " to " << range->second.second << ".\n";
                        continue;
                    }
                    row.push_back(value); break;
                } catch (...) { cout << "   Please type a number for " << name << " (or r for random).\n"; }
            }
        }
    }
    return row;
}

static void predictAllTargets(const vector<double>& row, const vector<string>& features) {
    ofstream txt("prediction_result.txt");
    ofstream js("prediction_result.json");
    if (!txt || !js) throw runtime_error("Could not create prediction output files.");
    cout << fixed << setprecision(4);
    txt << fixed << setprecision(4);
    printSample(txt, row, features); txt << "\nPredictions:\n";
    cout << "\nPredictions:\n";
    js << "{\n  \"predictions\": [\n";
    for (int t = 0; t < (int)TARGETS.size(); t++) {
        const string& target = TARGETS[t]; string prefix = "models/" + safeName(target);
        vector<double> labels; vector<string> savedFeatures = readSchemaNames(prefix + "_schema.txt", labels);
        if (savedFeatures != features) throw runtime_error("Feature order/schema differs for " + target + ". Retrain and save all targets.");
        RandomForest rf; XGBoostModel xgb; DecisionTree dt;
        if (!rf.load(prefix + "_rf.model") || !xgb.load(prefix + "_xgb.model") || !dt.load(prefix + "_dt.model"))
            throw runtime_error("Saved models missing for " + target + ". Choose option 1 and save models first.");
        ostringstream line; line << fixed << setprecision(4);
        js << "    {\"target\":\"" << jsonEscape(target) << "\", ";
        if (target == "Perpetrator Age") {
            double value = ensembleRegression(rf.predictValue(row), xgb.predictValue(row), dt.predictValue(row));
            line << setprecision(1) << value << " years";
            js << "\"predicted_value\":" << value << "}";
        } else {
            vector<double> p = ensembleProbabilities(rf.predictProba(row), xgb.predictProba(row), dt.predictProba(row));
            int pred = max_element(p.begin(), p.end()) - p.begin();
            int code = pred < (int)labels.size() ? (int)labels[pred] : pred;
            string name = enc.targetMaps.count(target) ? decodeName(enc.targetMaps[target], code) : to_string(code);
            line << name << "  (confidence " << p[pred] << ")";
            js << "\"prediction\":\"" << jsonEscape(name) << "\", \"confidence\":" << p[pred] << "}";
        }
        cout << "  " << left << setw(22) << target << ": " << line.str() << "\n";
        txt << "  " << left << setw(22) << target << ": " << line.str() << "\n";
        if (t + 1 < (int)TARGETS.size()) js << ",";
        js << "\n";
    }
    js << "  ]\n}\n";
    cout << "\nSaved prediction_result.txt and prediction_result.json\n";
}
static void predictionMenu(const string& datasetFile) {
    if (!encLoaded) throw runtime_error("encoding_mappings.json is needed for prediction. Put it beside the Makefile or pass it as the 2nd argument.");
    vector<double> labels; vector<string> features = readSchemaNames("models/Perpetrator_Sex_schema.txt", labels);
    cout << "\nPREDICTION MODE\n1. Enter a new sample manually\n2. Pick a random sample from the dataset\n3. Back\n";
    string choice = readLine("Choose: ");
    if (choice == "3") return;
    if (choice != "1" && choice != "2") { cout << "Invalid choice.\n"; return; }

    mt19937 rng((unsigned)chrono::system_clock::now().time_since_epoch().count());
    vector<vector<double>> rows;
    try { rows = readFeatureRows(datasetFile, features); } catch (const exception&) {}   // manual entry still works without the CSV
    vector<double> row;
    if (choice == "1") {
        row = enterSample(features, rows, rng);
        cout << "\n";
    } else {
        if (rows.empty()) throw runtime_error("No sample rows found in dataset.");
        int index = rng() % rows.size();
        row = rows[index];
        cout << "\nRandomly picked row " << index + 1 << " of " << rows.size() << " from the dataset.\n\n";
    }
    printSample(cout, row, features);
    predictAllTargets(row, features);
}

int main(int argc, char* argv[]) {
    cout << fixed << setprecision(4);
    string filename = "prepared_homicide.csv"; if (argc >= 2) filename = argv[1];
    string encodingFile = "encoding_mappings.json"; if (argc >= 3) encodingFile = argv[2];
    try {
        try { enc = loadEncoding(encodingFile); encLoaded = true; }
        catch (const exception& e) { cout << "Warning: " << e.what() << "\n"; }
    _mkdir("models");
        while (true) {
            cout << "\n========================================\n   HOMICIDE PREDICTION SYSTEM\n========================================\n";
            cout << "Dataset: " << filename << (encLoaded ? "   |   Mappings: " + encodingFile : "") << "\n\n";
            cout << "1. Train all five targets and evaluate models\n";
            cout << "2. Predict a case (type a sample or use a random dataset row)\n";
            cout << "3. Exit\n\nEnter your choice: ";
                        int choice = 0;
            string choiceText = readLine("");   // the prompt was already printed above
            if (choiceText == "1" || choiceText == "2" || choiceText == "3") choice = choiceText[0] - '0';
            if (choice == 3) { cout << "Goodbye.\n"; break; }
            if (choice == 1) {
                            string answer;
                while (true) {
                    answer = lower(readLine("Save/update learned model patterns in TXT and JSON for future predictions? (y/n): "));
                    if (answer == "y" || answer == "n") break;
                    cout << "Please type y or n.\n";
                }
                bool save = answer == "y";
                for (const string& target : TARGETS) trainTarget(filename, target, save);
            } else if (choice == 2) {
                predictionMenu(filename);
            } else cout << "Invalid choice. Please enter 1, 2, or 3.\n";
        }
    } catch (const exception& e) { cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}

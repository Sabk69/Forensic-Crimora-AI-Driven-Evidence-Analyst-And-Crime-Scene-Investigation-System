#include "common.h"
#include "models.h"
#include <direct.h>

static string safeName(const string& target) { string s = target; for (char& c : s) if (c == ' ') c = '_'; return s; }
static vector<string> readSchemaNames(const string& path, vector<double>& labels) {
    ifstream in(path); if (!in) throw runtime_error("Could not open model schema: " + path);
    int n; in >> n; vector<string> names(n);
    for (string& s : names) { in >> quoted(s); while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back(); }
    int count; in >> count; labels.resize(count); for (double& v : labels) in >> v;
    return names;
}
static vector<vector<double>> readFeatureCSV(const string& path, const vector<string>& names) {
    ifstream in(path); if (!in) throw runtime_error("Could not open input CSV: " + path);
    string line; getline(in, line); vector<string> header = splitCSV(line);
    map<string, int> pos; for (int i = 0; i < (int)header.size(); i++) pos[header[i]] = i;
    for (const string& name : names) if (!pos.count(name)) throw runtime_error("Missing feature column: " + name);
    vector<vector<double>> rows;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<string> values = splitCSV(line); if (values.size() != header.size()) continue;
        vector<double> row;
        for (const string& name : names) {
            double value = 0;
            try { value = stod(values[pos[name]]); } catch (...) { value = 0; }
            row.push_back(value);
        }
        rows.push_back(row);
    }
    return rows;
}
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: ./predict \"Perpetrator Sex\" new_cases.csv\n";
        return 1;
    }
    string target = argv[1], input = argv[2];
    bool valid = false; for (const string& t : TARGETS) if (t == target) valid = true;
    if (!valid) { cout << "Unknown target name.\n"; return 1; }
    try {
        string prefix = "models/" + safeName(target);
        vector<double> labels;
        vector<string> names = readSchemaNames(prefix + "_schema.txt", labels);
        vector<vector<double>> rows = readFeatureCSV(input, names);
        RandomForest rf; XGBoostModel xgb; DecisionTree dt;
        if (!rf.load(prefix + "_rf.model") || !xgb.load(prefix + "_xgb.model") || !dt.load(prefix + "_dt.model")) throw runtime_error("Could not load all three models. Train first.");
        cout << fixed << setprecision(4);
        cout << "Rows: " << rows.size() << "\nTarget: " << target << "\n\n";
        for (int i = 0; i < (int)rows.size(); i++) {
            cout << "Case " << i + 1 << ": ";
            if (target == "Perpetrator Age") {
                double prediction = ensembleRegression(rf.predictValue(rows[i]), xgb.predictValue(rows[i]), dt.predictValue(rows[i]));
                cout << "Predicted age = " << prediction << "\n";
            } else {
                vector<double> probabilities = ensembleProbabilities(rf.predictProba(rows[i]), xgb.predictProba(rows[i]), dt.predictProba(rows[i]));
                int predicted = max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
                if (predicted < (int)labels.size()) cout << "Prediction code = " << labels[predicted];
                else cout << "Prediction class = " << predicted;
                cout << " | Confidence = " << probabilities[predicted] << "\n";
            }
        }
    } catch (const exception& e) { cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}

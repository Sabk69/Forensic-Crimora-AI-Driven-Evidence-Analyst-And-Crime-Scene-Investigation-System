#include "common.h"

vector<string> splitCSV(const string& line) {
    vector<string> result;
    string value;
    bool quoted = false;
    for (int i = 0; i < (int)line.size(); i++) {
        char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < (int)line.size() && line[i + 1] == '"') {
                value += '"'; i++;
            } else quoted = !quoted;
        } else if (c == ',' && !quoted) {
            result.push_back(value); value.clear();
        } else value += c;
    }
    result.push_back(value);
    for (string& item : result) {
        while (!item.empty() && (item.back() == '\r' || item.back() == '\n')) item.pop_back();
    }
    return result;
}

static double toNumber(const string& s) {
    if (s.empty()) return 0.0;
    try { return stod(s); } catch (...) { return 0.0; }
}

Dataset readDataset(const string& filename, const string& targetName) {
    Dataset data;
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Could not open " + filename);
    string line;
    if (!getline(file, line)) throw runtime_error("CSV is empty");
    vector<string> headers = splitCSV(line);
    map<string, int> columns;
    for (int i = 0; i < (int)headers.size(); i++) columns[headers[i]] = i;
    if (!columns.count(targetName)) throw runtime_error("Target not found: " + targetName);
    vector<int> featureColumns;
    for (int i = 0; i < (int)headers.size(); i++) {
        bool target = false;
        for (const string& name : TARGETS) if (headers[i] == name) target = true;
        if (!target) {
            featureColumns.push_back(i);
            data.featureNames.push_back(headers[i]);
        }
    }
    int targetColumn = columns[targetName];
    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<string> values = splitCSV(line);
        if (values.size() != headers.size()) continue;
        double target = toNumber(values[targetColumn]);
        if (target < 0) continue;
        vector<double> row;
        for (int c : featureColumns) row.push_back(toNumber(values[c]));
        data.X.push_back(row);
        data.y.push_back(target);
    }
    return data;
}

void splitClassification(const vector<double>& y, vector<int>& train, vector<int>& test, unsigned seed) {
    mt19937 rng(seed);
    map<int, vector<int>> groups;
    for (int i = 0; i < (int)y.size(); i++) groups[(int)y[i]].push_back(i);
    for (auto& item : groups) {
        vector<int>& ids = item.second;
        shuffle(ids.begin(), ids.end(), rng);
        int count = (int)(ids.size() * 0.2);
        if (ids.size() >= 2 && count == 0) count = 1;
        if (count >= (int)ids.size()) count = (int)ids.size() - 1;
        for (int j = 0; j < (int)ids.size(); j++) {
            if (j < count) test.push_back(ids[j]);
            else train.push_back(ids[j]);
        }
    }
    shuffle(train.begin(), train.end(), rng);
    shuffle(test.begin(), test.end(), rng);
}

void splitRegression(int n, vector<int>& train, vector<int>& test, unsigned seed) {
    mt19937 rng(seed);
    vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = i;
    shuffle(ids.begin(), ids.end(), rng);
    int count = (int)(n * 0.2);
    for (int i = 0; i < n; i++) {
        if (i < count) test.push_back(ids[i]);
        else train.push_back(ids[i]);
    }
}

vector<int> makeClassLabels(const vector<double>& y, vector<double>& originalLabels) {
    set<int> unique;
    for (double value : y) unique.insert((int)value);
    originalLabels.clear();
    map<int, int> mapping;
    int next = 0;
    for (int value : unique) {
        mapping[value] = next++;
        originalLabels.push_back(value);
    }
    vector<int> result;
    for (double value : y) result.push_back(mapping[(int)value]);
    return result;
}

void classificationMetrics(const vector<int>& actual, const vector<int>& predicted, int classes,
                           const vector<string>& names, bool perClass) {
    if (actual.empty()) { cout << "No rows.\n"; return; }
    vector<vector<int>> cm(classes, vector<int>(classes, 0));
    int correct = 0;
    for (int i = 0; i < (int)actual.size(); i++) {
        if (actual[i] >= 0 && actual[i] < classes && predicted[i] >= 0 && predicted[i] < classes) cm[actual[i]][predicted[i]]++;
        if (actual[i] == predicted[i]) correct++;
    }
    double weighted = 0;
    if (perClass) cout << "Class-wise results:\n";
    for (int c = 0; c < classes; c++) {
        double tp = cm[c][c], actualCount = 0, predictedCount = 0;
        for (int j = 0; j < classes; j++) { actualCount += cm[c][j]; predictedCount += cm[j][c]; }
        double precision = predictedCount ? tp / predictedCount : 0;
        double recall = actualCount ? tp / actualCount : 0;
        double f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
        weighted += f1 * actualCount;
        if (perClass) {
            string label = c < (int)names.size() ? names[c] : "Class " + to_string(c);
            cout << left << setw(32) << label << right << " | Precision: " << precision << " | Recall: " << recall
                 << " | F1: " << f1 << " | Support: " << (int)actualCount << "\n";
        }
    }
    cout << "Accuracy: " << (double)correct / actual.size() << "\nWeighted F1: " << weighted / actual.size() << "\n";
}

void regressionMetrics(const vector<double>& actual, const vector<double>& predicted) {
    if (actual.empty()) { cout << "No test rows.\n"; return; }
    double mae = 0, mse = 0, mean = 0;
    for (double v : actual) mean += v;
    mean /= actual.size();
    double total = 0;
    for (int i = 0; i < (int)actual.size(); i++) {
        double e = actual[i] - predicted[i];
        mae += abs(e); mse += e * e; total += (actual[i] - mean) * (actual[i] - mean);
    }
    mae /= actual.size(); mse /= actual.size();
    cout << "\nMAE: " << mae << "\nRMSE: " << sqrt(mse) << "\nR2: " << (total > 0 ? 1 - mse * actual.size() / total : 0) << "\n";
}

// ---------- tiny JSON reader, only for the shape of encoding_mappings.json ----------
struct JsonReader {
    string s; size_t i = 0;
    void skip() { while (i < s.size() && isspace((unsigned char)s[i])) i++; }
    bool next(char c) { skip(); return i < s.size() && s[i] == c; }
    void expect(char c) {
        if (!next(c)) throw runtime_error(string("encoding_mappings.json: expected '") + c + "'");
        i++;
    }
    string text() {
        expect('"'); string r;
        while (i < s.size() && s[i] != '"') { if (s[i] == '\\') i++; r += s[i++]; }
        i++; return r;
    }
    double number() { skip(); char* end; double v = strtod(s.c_str() + i, &end); i = end - s.c_str(); return v; }
    vector<string> textList() {
        vector<string> r; expect('[');
        while (!next(']')) { r.push_back(text()); if (next(',')) i++; }
        i++; return r;
    }
    map<string, int> textToCode() {
        map<string, int> r; expect('{');
        while (!next('}')) { string k = text(); expect(':'); r[k] = (int)number(); if (next(',')) i++; }
        i++; return r;
    }
    map<string, map<string, int>> groups() {
        map<string, map<string, int>> r; expect('{');
        while (!next('}')) { string k = text(); expect(':'); r[k] = textToCode(); if (next(',')) i++; }
        i++; return r;
    }
};

Encoding loadEncoding(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Could not open " + path + " (put it beside the Makefile or pass it as the 2nd argument)");
    JsonReader r; r.s.assign((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    Encoding e; r.expect('{');
    while (!r.next('}')) {
        string key = r.text(); r.expect(':');
        if (key == "input_columns") e.inputColumns = r.textList();
        else if (key == "targets") r.textList();
        else if (key == "categorical_feature_mappings") e.featureMaps = r.groups();
        else if (key == "target_mappings") e.targetMaps = r.groups();
        else if (key == "numeric_medians") {
            r.expect('{');
            while (!r.next('}')) { string k = r.text(); r.expect(':'); e.medians[k] = r.number(); if (r.next(',')) r.i++; }
            r.i++;
        } else throw runtime_error("encoding_mappings.json: unknown section " + key);
        if (r.next(',')) r.i++;
    }
    return e;
}

string decodeName(const map<string, int>& mapping, int code) {
    for (const auto& item : mapping) if (item.second == code) return item.first;
    return to_string(code);
}

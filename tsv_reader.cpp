#include "tsv_reader.h" // declarations
#include <fstream> // ifstream
#include <sstream> // getline
#include <stdexcept> // runtime_error

static bool split_two_fields(const std::string& line, std::string& a, std::string& b) { // split at first tab
  auto pos = line.find('\t'); // find tab
  if (pos == std::string::npos) return false; // invalid line
  a = line.substr(0, pos); // first field
  b = line.substr(pos + 1); // second field
  return true; // ok
}

std::vector<TrainRow> read_train_tsv(const std::string& path) {
  std::ifstream in(path); // open file
  if (!in) throw std::runtime_error("cannot open train.tsv: " + path); // fail early

  std::string header; std::getline(in, header); // skip header

  std::vector<TrainRow> rows; // output
  std::string line; // buffer
  while (std::getline(in, line)) { // read line
    std::string a, b; // fields
    if (!split_two_fields(line, a, b)) continue; // skip malformed
    TrainRow r; // row
    r.label = std::stoi(a); // parse label
    r.review = b; // review
    rows.push_back(std::move(r)); // store
  }
  return rows; // return all rows
}

std::vector<TestRow> read_test_tsv(const std::string& path) {
  std::ifstream in(path); // open file
  if (!in) throw std::runtime_error("cannot open test.tsv: " + path); // fail early

  std::string header; std::getline(in, header); // skip header

  std::vector<TestRow> rows; // output
  std::string line; // buffer
  while (std::getline(in, line)) { // read line
    std::string a, b; // fields
    if (!split_two_fields(line, a, b)) continue; // skip malformed
    TestRow r; // row
    r.id = std::stoi(a); // parse id
    r.review = b; // review
    rows.push_back(std::move(r)); // store
  }
  return rows; // return all rows
}
#include "csv_write.h" // decl
#include <fstream> // ofstream
#include <stdexcept> // runtime_error

void write_submission_csv(const std::string& path, const std::vector<int>& ids, const std::vector<int>& labels) {
  if (ids.size() != labels.size()) throw std::runtime_error("ids/labels size mismatch"); // guard
  std::ofstream out(path); // open text
  if (!out) throw std::runtime_error("cannot write csv: " + path); // fail
  out << "id,label\n"; // required header
  for (size_t i = 0; i < ids.size(); i++) out << ids[i] << "," << labels[i] << "\n"; // rows
}
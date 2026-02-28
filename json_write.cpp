#include "json_write.h" // decl
#include <fstream> // ofstream
#include <stdexcept> // runtime_error

void write_metrics_json(const std::string& path, float train_acc, float valid_acc) {
  std::ofstream out(path); // open file
  if (!out) throw std::runtime_error("cannot write json: " + path); // fail
  out << "{\n"; // json open
  out << "  \"train_accuracy\": " << train_acc << ",\n"; // metric
  out << "  \"valid_accuracy\": " << valid_acc << "\n"; // metric
  out << "}\n"; // json close
}
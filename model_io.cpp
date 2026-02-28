#include "model_io.h" // decl
#include <fstream> // fstream
#include <stdexcept> // runtime_error

void save_model_bin(const LogRegModel& m, const std::string& path) {
  std::ofstream out(path, std::ios::binary); // open binary
  if (!out) throw std::runtime_error("cannot write model: " + path); // fail
  out.write(reinterpret_cast<const char*>(&m.dim), sizeof(int)); // dim
  out.write(reinterpret_cast<const char*>(&m.b), sizeof(float)); // bias
  out.write(reinterpret_cast<const char*>(m.w.data()), sizeof(float) * m.w.size()); // weights
}

LogRegModel load_model_bin(const std::string& path) {
  std::ifstream in(path, std::ios::binary); // open binary
  if (!in) throw std::runtime_error("cannot read model: " + path); // fail

  LogRegModel m; // model
  in.read(reinterpret_cast<char*>(&m.dim), sizeof(int)); // dim
  in.read(reinterpret_cast<char*>(&m.b), sizeof(float)); // bias
  m.w.resize(m.dim); // allocate weights
  in.read(reinterpret_cast<char*>(m.w.data()), sizeof(float) * m.w.size()); // weights
  return m;
}
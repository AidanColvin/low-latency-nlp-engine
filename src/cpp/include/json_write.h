#pragma once
#include <string> // string

void write_metrics_json(const std::string& path, float train_acc, float valid_acc); // write small json

// -------------------------------
// COMPAT SHIM: JsonWriter
// Header-only minimal JSON writer (kv + close).
// -------------------------------
#include <fstream>
#include <string>

struct JsonWriter {
  std::ofstream out;
  bool first = true;

  explicit JsonWriter(const std::string& path) : out(path) { out << "{\n"; }

  static std::string esc(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
      if (c == '\\') r += "\\\\";
      else if (c == '"') r += "\\\"";
      else if (c == '\n') r += "\\n";
      else r += c;
    }
    return r;
  }

  void comma() {
    if (!first) out << ",\n";
    first = false;
  }

  void kv(const std::string& k, const std::string& v) {
    comma();
    out << "  \"" << esc(k) << "\": \"" << esc(v) << "\"";
  }
  void kv(const std::string& k, int v) {
    comma();
    out << "  \"" << esc(k) << "\": " << v;
  }
  void kv(const std::string& k, double v) {
    comma();
    out << "  \"" << esc(k) << "\": " << v;
  }
  void kv(const std::string& k, bool v) {
    comma();
    out << "  \"" << esc(k) << "\": " << (v ? "true" : "false");
  }

  void close() {
    out << "\n}\n";
    out.close();
  }
};

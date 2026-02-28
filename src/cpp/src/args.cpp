#include "args.h" // parser api
#include <cstdlib> // std::atoi std::atof

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) { // --k v pairs
  std::unordered_map<std::string, std::string> a; // args
  for (int i = 1; i < argc; ++i) { // iterate argv
    std::string key = argv[i]; // token
    if (key.rfind("--", 0) == 0) { // starts with --
      std::string name = key.substr(2); // strip --
      std::string val = "1"; // default flag value
      if (i + 1 < argc) { // lookahead
        std::string nxt = argv[i + 1]; // next token
        if (nxt.rfind("--", 0) != 0) { val = nxt; ++i; } // consume value
      }
      a[name] = val; // store
    }
  }
  return a; // parsed args
}

std::string get_arg(const std::unordered_map<std::string, std::string>& a, const std::string& k, const std::string& d) { // string get
  auto it = a.find(k); // lookup
  return (it == a.end()) ? d : it->second; // default if missing
}

int get_int(const std::unordered_map<std::string, std::string>& a, const std::string& k, int d) { // int get
  auto it = a.find(k); // lookup
  return (it == a.end()) ? d : std::atoi(it->second.c_str()); // parse int
}

double get_double(const std::unordered_map<std::string, std::string>& a, const std::string& k, double d) { // double get
  auto it = a.find(k); // lookup
  return (it == a.end()) ? d : std::atof(it->second.c_str()); // parse double
}

#include "tokenizer.h" // declaration

std::vector<std::string> split_whitespace(const std::string& s) {
  std::vector<std::string> toks; // output
  std::string cur; // current token
  for (char c : s) { // iterate chars
    if (c == ' ') { // delimiter
      if (!cur.empty()) { toks.push_back(cur); cur.clear(); } // flush token
    } else {
      cur.push_back(c); // build token
    }
  }
  if (!cur.empty()) toks.push_back(cur); // final flush
  return toks; // return tokens
}
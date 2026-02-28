#include "text_clean.h" // declaration
#include <cctype> // std::tolower, std::isspace

std::string clean_text_ascii(const std::string& s) {
  std::string out; out.reserve(s.size()); // pre-allocate
  bool prev_space = true; // collapse spaces

  for (unsigned char ch : s) { // iterate bytes
    char c = static_cast<char>(ch); // char
    if (std::isalnum(ch)) { // keep letters/digits
      out.push_back(static_cast<char>(std::tolower(ch))); // lowercase
      prev_space = false; // not space
    } else {
      if (!prev_space) { out.push_back(' '); prev_space = true; } // convert punctuation to single space
    }
  }

  if (!out.empty() && out.back() == ' ') out.pop_back(); // trim trailing space
  return out; // normalized text
}
#include "hashing.h" // declarations

uint64_t fnv1a_64(const std::string& s) {
  uint64_t h = 1469598103934665603ULL; // FNV offset basis
  for (unsigned char c : s) { // each byte
    h ^= static_cast<uint64_t>(c); // xor
    h *= 1099511628211ULL; // multiply prime
  }
  return h; // hash
}

int hash_to_dim(uint64_t h, int dim) {
  return static_cast<int>(h % static_cast<uint64_t>(dim)); // modulo
}

int hash_sign(uint64_t h) {
  return (h & 1ULL) ? 1 : -1; // odd -> +1, even -> -1
}
#pragma once
#include <string> // std::string
#include <cstdint> // uint64_t

uint64_t fnv1a_64(const std::string& s); // stable hash
int hash_to_dim(uint64_t h, int dim); // map hash to [0, dim)
int hash_sign(uint64_t h); // +1 or -1 for signed hashing
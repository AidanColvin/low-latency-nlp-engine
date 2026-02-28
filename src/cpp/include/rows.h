#pragma once // header guard

#include <string> // std::string

struct TrainRow { // one training example
  int label;      // 0 or 1
  std::string review; // raw review text
};

struct TestRow { // one test example
  int id;         // 1..6000
  std::string review; // raw review text
};

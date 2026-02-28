#pragma once

#include "rows.h" // TrainRow/TestRow
#include <string> // std::string
#include <vector> // std::vector

std::vector<TrainRow> read_train_tsv(const std::string& path); // reads label \t review
std::vector<TestRow>  read_test_tsv (const std::string& path); // reads id \t review
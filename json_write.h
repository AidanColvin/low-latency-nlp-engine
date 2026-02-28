#pragma once
#include <string> // string

void write_metrics_json(const std::string& path, float train_acc, float valid_acc); // write small json
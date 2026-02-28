#pragma once
#include <torch/script.h>
#include <string>

/*
given a file path string
return loaded torch module
throws error if loading fails
*/
torch::jit::script::Module load_traced_model(const std::string& file_path);

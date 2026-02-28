#include "tb_loader.hpp"
#include <iostream>

torch::jit::script::Module load_traced_model(const std::string& file_path) {
    try {
        return torch::jit::load(file_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model." << std::endl;
        throw;
    }
}

#include <iostream>
#include "tb_config.hpp"
#include "tb_loader.hpp"

/*
given nothing
return integer exit status
loads tinybert model into memory
*/
int main() {
    torch::jit::script::Module module = load_traced_model(EXPORT_FILE);
    std::cout << "Model loaded successfully." << std::endl;
    return 0;
}

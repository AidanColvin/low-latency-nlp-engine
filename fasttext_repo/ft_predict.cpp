#include "ft_predict.hpp"
#include <iostream>
#include <sstream>
#include <vector>

void predict_label(fasttext::FastText& model, const std::string& text) {
    std::istringstream in(text);
    std::vector<std::pair<fasttext::real, std::string>> predictions;
    model.predictLine(in, predictions, 1, 0.0);
    
    for (const auto& pred : predictions) {
        std::cout << "Label: " << pred.second << " Confidence: " << pred.first << std::endl;
    }
}

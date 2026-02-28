#pragma once
#include <string>
#include "fasttext.h"

/*
given a fasttext model reference and input string
return nothing
prints predicted label to standard output
*/
void predict_label(fasttext::FastText& model, const std::string& text);

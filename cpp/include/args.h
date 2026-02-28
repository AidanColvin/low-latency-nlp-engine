#pragma once // minimal cli args parser

#include <string> // std::string
#include <unordered_map> // map

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv); // parses --key value
std::string get_arg(const std::unordered_map<std::string, std::string>& a, const std::string& k, const std::string& d); // defaulted get
int get_int(const std::unordered_map<std::string, std::string>& a, const std::string& k, int d); // int get
double get_double(const std::unordered_map<std::string, std::string>& a, const std::string& k, double d); // double get

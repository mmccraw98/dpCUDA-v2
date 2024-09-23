#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "orchestrator.h"
#include "../particle/particle.h"

 // TODO: move all non-file related functionality to iomanager

std::ifstream open_input_file(std::string file_name);
std::ofstream open_output_file(std::string file_name, bool overwrite = false);
void make_dir(const std::string& dir_name, bool overwrite = false);
bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings);
long get_largest_file_index(std::string dir_name, std::string file_prefix = "");

template <typename T>
void write_array_to_file(const std::string& file_name, const thrust::host_vector<T>& data, long num_rows, long num_cols, int precision);

template <typename T>
thrust::host_vector<T> read_array_from_file(const std::string& file_name, long num_rows, long num_cols);

// make a python resume script that loads the configuration, gets the script details and arguments, and calls the relevant script with optionally overwriting some arguments

#endif /* IO_UTILS_H */
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
#include <nlohmann/json.hpp>

 // TODO: move all non-file related functionality to iomanager

std::ifstream open_input_file(std::string file_name);
std::ofstream open_output_file(std::string file_name, bool overwrite = false);
void make_dir(const std::string& dir_name, bool overwrite = false);
bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings);
long get_largest_file_index(std::string dir_name, std::string file_prefix = "");
void write_json_to_file(const std::string& file_name, const nlohmann::json& data);

template <typename T>
void write_array_to_file(const std::string& file_name, const thrust::host_vector<T>& data, long num_rows, long num_cols, int precision) {
    std::ofstream output_file(file_name);
    if (!output_file.is_open()) {
        std::cerr << "write_array_to_file: Error: could not open output file " << file_name << std::endl;
        exit(1);
    }

    for (long row = 0; row < num_rows; row++) {
        for (long col = 0; col < num_cols; col++) {
            output_file << std::setprecision(precision) << data[row * num_cols + col] << "\t";
        }
        output_file << std::endl;
    }

    output_file.close();
}


template <typename T>
thrust::host_vector<T> read_array_from_file(const std::string& file_name, long num_rows, long num_cols) {
    thrust::host_vector<T> data(num_rows * num_cols);
    std::ifstream input_file(file_name);
    if (!input_file.is_open()) {
        std::cerr << "read_array_from_file: Error: could not open input file " << file_name << std::endl;
        exit(1);
    }

    std::string input_string;
    for (long row = 0; row < num_rows; row++) {
        for (long col = 0; col < num_cols; col++) {
            if (!(input_file >> data[row * num_cols + col])) {
                std::cerr << "Error: insufficient data in file " << file_name << std::endl;
                exit(1);
            }
        }
    }

    input_file.close();
    return data;
}



// make a python resume script that loads the configuration, gets the script details and arguments, and calls the relevant script with optionally overwriting some arguments

#endif /* IO_UTILS_H */
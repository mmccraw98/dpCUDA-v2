#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/utils.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

std::ifstream open_input_file(std::string file_name) {
    std::ifstream input_file = std::ifstream(file_name.c_str());
    if (!input_file.is_open()) {
        std::cerr << "ERROR: open_input_file: could not open: " << file_name << std::endl;
        exit(1);
    }
    return input_file;
}

std::ofstream open_output_file(std::string file_name, bool overwrite) {
    std::ofstream output_file;
    if (overwrite) {
        output_file.open(file_name.c_str(), std::ios::out | std::ios::trunc);
    } else {
        output_file.open(file_name.c_str(), std::ios::out | std::ios::app);
    }

    if (!output_file.is_open()) {
        std::cerr << "ERROR: open_output_file: could not open: " << file_name << std::endl;
        exit(1);
    }
    return output_file;
}

void make_dir(const std::string& dir_name, bool overwrite) {
    if (std::filesystem::exists(dir_name)) {
        if (!overwrite) {
            // std::cerr << "ERROR: make_dir: directory exists: " << dir_name << std::endl;
            // exit(1);
            return;
        }
    } else {
        try {
            std::filesystem::create_directories(dir_name);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "ERROR: make_dir: issue creating directory: " << dir_name << " Error code: " << e.code() << std::endl;
            exit(1);
        }
    }
}


bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings) {
    for (const auto& substring : substrings) {
        if (string.find(substring) != std::string::npos) {
            return true;
        }
    }
    return false;
}

long get_largest_file_index(std::string dir_name, std::string file_prefix) {
    long largest_index = -1;
    for (const auto & entry : std::filesystem::directory_iterator(dir_name)) {
        std::string current_file = entry.path().filename().string();
        if (current_file.find(file_prefix) == file_prefix.length()) {
            long step = std::stol(current_file.substr(file_prefix.length()));
            if (step > largest_index) {
                largest_index = step;
            }
        }
    }
    if (largest_index == -1) {
        std::cout << "ERROR: get_largest_file_index: could not find any indexed files in " << dir_name << " with matching prefix '" << file_prefix << "'" << std::endl;
        exit(1);
    }
    return largest_index;
}

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



// template <typename T>
// void writeKeyValToParams(string dirName, string key, T value) {
//     string fileParams = dirName + "params.dat";
//     ofstream saveParams(fileParams.c_str(), ios::app);  // Open in append mode
//     if (!saveParams.is_open()) {
//         cerr << "Error opening file: " << fileParams << endl;
//         return;
//     }
//     saveParams << key << "\t" << value << endl;
//     saveParams.close();
// }
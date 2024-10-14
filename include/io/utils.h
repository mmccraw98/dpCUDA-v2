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

/**
 * @brief Opens an input file
 * @param file_name The name of the file to open
 * @return The input file stream
 */
std::ifstream open_input_file(std::string file_name);

/**
 * @brief Opens an output file
 * @param file_name The name of the file to open
 * @param overwrite Whether to overwrite the file if it already exists
 * @return The output file stream
 */
std::ofstream open_output_file(std::string file_name, bool overwrite = false);

/**
 * @brief Makes a directory
 * @param dir_name The name of the directory to make
 * @param overwrite Whether to overwrite the directory if it already exists
 */
void make_dir(const std::string& dir_name, bool overwrite = false);

/**
 * @brief Checks if a string contains any of the substrings
 * @param string The string to check
 * @param substrings The substrings to check for
 * @return Whether the string contains any of the substrings
 */
bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings);

/**
 * @brief Gets the largest file index in a directory
 * @param dir_name The name of the directory to check
 * @param file_prefix The prefix of the files to check for
 * @return The largest file index
 */
long get_largest_file_index(std::string dir_name, std::string file_prefix = "");

/**
 * @brief Writes a JSON object to a file
 * @param file_name The name of the file to write to
 * @param data The JSON object to write
 */
void write_json_to_file(const std::string& file_name, const nlohmann::json& data);

/**
 * @brief Writes an array to a file
 * @param file_name The name of the file to write to
 * @param data The array to write
 * @param num_rows The number of rows in the array
 * @param num_cols The number of columns in the array
 * @param precision The number of decimal places to write
 */
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

/**
 * @brief Reads an array from a file
 * @param file_name The name of the file to read from
 * @param num_rows The number of rows in the array
 * @param num_cols The number of columns in the array
 * @return The array read from the file
 */
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

#endif /* IO_UTILS_H */
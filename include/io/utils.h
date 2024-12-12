#pragma once

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
#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

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

// Helper function to access the correct data type from the variant
template <typename T>
thrust::host_vector<T>& get_1d_data(ArrayData& array_data) {
    return std::get<thrust::host_vector<T>>(array_data.data);
}

template <typename T>
std::pair<thrust::host_vector<T>, thrust::host_vector<T>>& get_2d_data(ArrayData& array_data) {
    return std::get<std::pair<thrust::host_vector<T>, thrust::host_vector<T>>>(array_data.data);
}

void reorder_array(ArrayData& array_data, const ArrayData& reorder_index_data);

template <typename T>
void write_1d_array_to_file(std::ofstream& output_file, const thrust::host_vector<T>& data, int precision) {
    for (const auto& val : data) {
        output_file << std::setprecision(precision) << val << "\n";
    }
}

template <typename T>
void write_2d_array_to_file(std::ofstream& output_file, 
                            const std::pair<thrust::host_vector<T>, thrust::host_vector<T>>& data, 
                            int precision) {
    const auto& first_vector = data.first;
    const auto& second_vector = data.second;

    if (first_vector.size() != second_vector.size()) {
        std::cerr << "Error: 2D data vectors have mismatched sizes." << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < first_vector.size(); ++i) {
        output_file << std::setprecision(precision) 
                    << first_vector[i] << "\t" 
                    << second_vector[i] << "\n";
    }
}

void write_array_data_to_file(const std::string& file_name, ArrayData& array_data, long precision);

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

template <typename T>
Data1D<T> read_1d_data_from_file(const std::string& file_name, long num_rows) {
    auto data = read_array_from_file<T>(file_name, num_rows, 1);
    Data1D<T> data_1d;
    data_1d.resize(num_rows);
    data_1d.setData(data);
    return data_1d;
}

template <typename T>
SwapData1D<T> read_1d_swap_data_from_file(const std::string& file_name, long num_rows) {
    auto data = read_array_from_file<T>(file_name, num_rows, 1);
    SwapData1D<T> data_1d;
    data_1d.resize(num_rows);
    data_1d.setData(data);
    return data_1d;
}

template <typename T>
Data2D<T> read_2d_data_from_file(const std::string& file_name, long num_rows, long num_cols) {
    auto data = read_array_from_file<T>(file_name, num_rows, num_cols);
    Data2D<T> data_2d;
    data_2d.resize(num_rows);
    thrust::host_vector<T> x_data(num_rows);
    thrust::host_vector<T> y_data(num_rows);
    
    // Deinterleave the data
    for(long i = 0; i < num_rows; i++) {
        x_data[i] = data[i * 2];
        y_data[i] = data[i * 2 + 1];
    }
    
    data_2d.setData(x_data, y_data);
    return data_2d;
}

template <typename T>
SwapData2D<T> read_2d_swap_data_from_file(const std::string& file_name, long num_rows, long num_cols) {
    auto data = read_array_from_file<T>(file_name, num_rows, num_cols);
    SwapData2D<T> data_2d;
    data_2d.resize(num_rows);
    thrust::host_vector<T> x_data(num_rows);
    thrust::host_vector<T> y_data(num_rows);
    
    // Deinterleave the data
    for(long i = 0; i < num_rows; i++) {
        x_data[i] = data[i * 2];
        y_data[i] = data[i * 2 + 1];
    }
    
    data_2d.setData(x_data, y_data);
    return data_2d;
}
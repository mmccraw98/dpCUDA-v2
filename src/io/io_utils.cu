#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/io/io_utils.h"
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
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <variant>
#include <string>
#include <utility>

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

std::filesystem::path get_path(std::filesystem::path root_path, std::string name) {
    std::filesystem::path path = root_path / name;
    if (!std::filesystem::exists(path)) {
        std::cerr << "ERROR: get_path: path does not exist: " << path << std::endl;
        exit(1);
    }
    return path;
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
        if (current_file.find(file_prefix) == 0) {
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

std::tuple<std::filesystem::path, long> get_trajectory_frame_path(std::filesystem::path trajectory_root, std::string file_prefix, long file_index) {
    std::filesystem::path best_path;
    long best_frame = -1;

    // Single pass through the directory:
    // • If file_index == -1, track the largest frame encountered
    // • Otherwise, break early if we find an exact matching frame index.
    for (const auto& entry : std::filesystem::directory_iterator(trajectory_root)) {
        const auto filename = entry.path().filename().string();

        // Ensure the file actually starts with the prefix
        if (filename.rfind(file_prefix, 0) == 0) {
            // Extract the numeric part after the prefix
            const long frame_val = std::stol(filename.substr(file_prefix.size()));

            // If file_index == -1, we want the largest frame
            if (file_index == -1) {
                if (frame_val > best_frame) {
                    best_frame = frame_val;
                    best_path = entry.path();
                }
            }
            // Otherwise, if we find an exact match, return immediately
            else {
                if (frame_val == file_index) {
                    return std::make_tuple(entry.path(), frame_val);
                }
            }
        }
    }

    // If we wanted the largest frame (file_index == -1), return what we found
    if (file_index == -1 && !best_path.empty()) {
        return std::make_tuple(best_path, best_frame);
    }

    // Otherwise, we did not find a matching path
    std::cerr << "ERROR: get_trajectory_frame_path: no matching frame path for prefix '"
              << file_prefix << "' and index " << file_index
              << " in " << trajectory_root << std::endl;
    exit(1);
}

void write_json_to_file(const std::string& file_name, const nlohmann::json& data) {
    std::ofstream output_file = open_output_file(file_name, true);
    output_file << data.dump(4);
    output_file.close();
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




// Reordering 1D array
void reorder_array_1d(ArrayData& array_data, const ArrayData& reorder_index_data) {
    if (array_data.type == DataType::Double) {
        auto& data = get_1d_data<double>(array_data);
        auto& index = get_1d_data<long>(const_cast<ArrayData&>(reorder_index_data));
        thrust::host_vector<double> new_data(array_data.size[0]);

        thrust::scatter(data.begin(), data.end(), index.begin(), new_data.begin());
        thrust::swap(data, new_data);
    } else if (array_data.type == DataType::Long) {
        auto& data = get_1d_data<long>(array_data);
        auto& index = get_1d_data<long>(const_cast<ArrayData&>(reorder_index_data));
        thrust::host_vector<long> new_data(array_data.size[0]);

        thrust::scatter(data.begin(), data.end(), index.begin(), new_data.begin());
        thrust::swap(data, new_data);
    }
}

// Reordering 2D array
void reorder_array_2d(ArrayData& array_data, const ArrayData& reorder_index_data) {
    if (array_data.type == DataType::Double) {
        auto& data = get_2d_data<double>(array_data);
        auto& index = get_1d_data<long>(const_cast<ArrayData&>(reorder_index_data));
        thrust::host_vector<double> new_first(data.first.size());
        thrust::host_vector<double> new_second(data.second.size());

        thrust::scatter(data.first.begin(), data.first.end(), index.begin(), new_first.begin());
        thrust::scatter(data.second.begin(), data.second.end(), index.begin(), new_second.begin());

        thrust::swap(data.first, new_first);
        thrust::swap(data.second, new_second);
    } else if (array_data.type == DataType::Long) {
        auto& data = get_2d_data<long>(array_data);
        auto& index = get_1d_data<long>(const_cast<ArrayData&>(reorder_index_data));
        thrust::host_vector<long> new_first(data.first.size());
        thrust::host_vector<long> new_second(data.second.size());

        thrust::scatter(data.first.begin(), data.first.end(), index.begin(), new_first.begin());
        thrust::scatter(data.second.begin(), data.second.end(), index.begin(), new_second.begin());

        thrust::swap(data.first, new_first);
        thrust::swap(data.second, new_second);
    }
}

// General function to reorder array based on dimensionality
void reorder_array(ArrayData& array_data, const ArrayData& reorder_index_data) {
    if (array_data.size[1] == 1) {
        reorder_array_1d(array_data, reorder_index_data);
    } else if (array_data.size[1] == 2) {
        reorder_array_2d(array_data, reorder_index_data);
    } else {
        std::cerr << "ERROR: reorder_array: array_data has invalid number of dimensions: " 
                  << array_data.size[1] << std::endl;
        exit(1);
    }
}

void write_array_data_to_file(const std::string& file_name, ArrayData& array_data, long precision) {
    std::ofstream output_file(file_name);
    if (!output_file.is_open()) {
        std::cerr << "write_array_data_to_file: Error: could not open output file " << file_name << std::endl;
        exit(1);
    }

    if (array_data.size[1] == 1) {
        if (array_data.type == DataType::Double) {
            auto& data = get_1d_data<double>(array_data);
            write_1d_array_to_file(output_file, data, precision);
        } else if (array_data.type == DataType::Long) {
            auto& data = get_1d_data<long>(array_data);
            write_1d_array_to_file(output_file, data, precision);
        }
    } else if (array_data.size[1] == 2) {
        if (array_data.type == DataType::Double) {
            auto& data = get_2d_data<double>(array_data);
            write_2d_array_to_file(output_file, data, precision);
        } else if (array_data.type == DataType::Long) {
            auto& data = get_2d_data<long>(array_data);
            write_2d_array_to_file(output_file, data, precision);
        }
    } else {
        std::cerr << "write_array_data_to_file: Error: Unsupported array dimensionality: " 
                  << array_data.size[1] << std::endl;
        exit(1);
    }

    output_file.close();
}

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/orchestrator.h"  
#include "../../include/io/file_manager.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
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

std::ifstream open_input_file(std::string fname) {
    std::ifstream input_file = ifstream(file_name.c_str());
    if (!input_file.is_open()) {
        std::cerr << "ERROR: open_input_file: could not open: " << fname << std::endl;
        exit(1);
    }
    return input_file;
}

std::ofstream open_output_file(std::string fname) {
    std::ofstream output_file = ofstream(file_name.c_str());
    if (!output_fule.is_open()) {
        cerr << "ERROR: open_output_file: could not open: " << fname << std::endl;
        exit(1);
    }
    return output_file;
}

void make_dir(const std::string& dir_name, bool warn = true) {
    if (std::filesystem::exists(dir_name)) {
        if (warn) {
            std::cerr << "ERROR: make_dir: directory exists: " << dir_name << std::endl;
            exit(1);
        }
    } else {
        try {
            std::filesystem::create_directories(dir_name);
        } catch (const std::filesystem::flesystem_error& e) {
            std::cerr << "ERROR: make_dir: issue creating directory: " << dir_name << " Error code: " << e << std::endl;
            exit(1);
        }
    }
}


bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings) {
    for (const auto& substring : substrings) {
        if (inputString.find(substring) != std::string::npos) {
            return true;
        }
    }
    return false;
}

long get_largest_file_index(std::string dir_name, std::string file_prefix) {
    long largest_index = -1;
    for (const auto & entry : std::experimental::filesystem::directory_iterator(dir_name)) {
        std::string current_file = entry.path().filename().string();
        if (current_file.find(file_prefix) == len(file_prefix)) {  // TODO: fix the len(file_prefix)
            long step = std::stol(current_file.substr(len(file_prefix))); // TODO: fix the len(file_prefix)
            if (step > max_step) {
                max_step = step;
            }
        }
    }
    if (largest_index == -1) {
        std::cout << "ERROR: get_largest_file_index: could not find any indexed files in " << dir_name << " with matching prefix '" << file_prefix << "'" << std::endl;
    }
    return max_step;
}


FileManager::FileManager(Particle& particle, const std::vector<std::string>& log_names) : energy_orchestrator(particle, log_names) {
    std::cout << "FileManager::FileManager: Start" << std::endl;
    std::cout << "FileManager::FileManager: End" << std::endl;
}

FileManager::~FileManager() {
    std::cout << "FileManager::~FileManager: Start" << std::endl;
    std::cout << "FileManager::~FileManager: End" << std::endl;
}

void FileManager::write_energy_header() {
    // TODO: change this to work with resume
    long num_names = energy_orchestrator.log_names.size();
    std::cout << std::setw(width) << "step" << delimeter;
    for (long i = 0; i < num_names; i++) {
        std::cout << std::setw(width) << energy_orchestrator.log_names[i];
        if (i < num_names - 1) {
            std::cout << delimeter;
        }
    }

    std::cout << std::endl;
}

void FileManager::write_energy_values(long step) {
    energy_orchestrator.precalculate();
    if (!energy_file_has_header) {write_header();}
    std::cout << std::setw(width) << step << delimeter << std::setw(width);
    for (long i = 0; i < orchestrator.log_names.size(); i++) {
        double value = orchestrator.get_value(orchestrator.log_names[i]);
        value = orchestrator.apply_modifier(orchestrator.log_names[i], value);
        std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << value;
        if (i < orchestrator.log_names.size() - 1) {
            std::cout << delimeter;
        }
    }
    std::cout << std::endl;
}


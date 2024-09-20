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

FileManager::FileManager(Particle& particle, const std::vector<std::string>& log_names) : orchestrator(particle, log_names) {
    std::cout << "FileManager::FileManager: Start" << std::endl;
    std::cout << "FileManager::FileManager: End" << std::endl;
}

FileManager::~FileManager() {
    std::cout << "FileManager::~FileManager: Start" << std::endl;
    std::cout << "FileManager::~FileManager: End" << std::endl;
}

void FileManager::write_header() {
    long num_names = orchestrator.log_names.size();
    long total_width = (width + 3) * (num_names + 1) - 1;
    std::cout << std::string(total_width, '_') << std::endl;
    std::cout << std::setw(width) << "step" << " | ";
    
    for (long i = 0; i < num_names; i++) {
        std::cout << std::setw(width) << orchestrator.log_names[i];
        if (i < num_names - 1) {
            std::cout << " | ";
        }
    }

    std::cout << std::endl;
    std::cout << std::string(total_width, '_') << std::endl;
}

void FileManager::write_values(long step) {
    orchestrator.precalculate();
    if (last_header_log_step >= header_log_step_frequency) {
        write_header();
        last_header_log_step = 0;
    } else {
        last_header_log_step += 1;
    }
    std::cout << std::setw(width) << step << " | " << std::setw(width);
    for (long i = 0; i < orchestrator.log_names.size(); i++) {
        double value = orchestrator.get_value(orchestrator.log_names[i]);
        value = orchestrator.apply_modifier(orchestrator.log_names[i], value);
        std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << value;
        if (i < orchestrator.log_names.size() - 1) {
            std::cout << " | ";
        }
    }
    std::cout << std::endl;
}
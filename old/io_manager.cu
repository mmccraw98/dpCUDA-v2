#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/orchestrator.h"  
#include "../../include/io/step_manager.h"
#include "../../include/io/file_manager.h"
#include "../../include/io/io_manager.h"
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


FileManager::FileManager(Particle& particle, FileManagerConfig& config) {
    std::cout << "FileManager::FileManager: Start" << std::endl;
    this->particle = particle;
    this->config = config;




    std::cout << "FileManager::FileManager: End" << std::endl;
}

FileManager::~FileManager() {
    std::cout << "FileManager::~FileManager: Start" << std::endl;
    std::cout << "FileManager::~FileManager: End" << std::endl;
}


void FileManager::initialize_step_managers() {
    // this->energy_step_manager = StepManager(config.energy_save_style, config.energy_save_frequency, config.min_save_decade);

    // define the energy 

}



void FileManager::orchestrate(long step) {
    if (step % config.energy_save_frequency == 0) {
        write_energy_values(step);
    }
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


#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/logger.h"
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

Logger::Logger(Particle& particle, const std::vector<std::string>& log_names) : particle(particle), log_names(log_names) {
    std::cout << "Logger::Logger: Start" << std::endl;
    std::cout << "Logger::Logger: End" << std::endl;
    
}

Logger::~Logger() {
    std::cout << "Logger::~Logger: Start" << std::endl;
    std::cout << "Logger::~Logger: End" << std::endl;
}

void Logger::write_header(long width) {
    long num_names = log_names.size();
    long total_width = (width + 3) * (num_names + 1) - 1;
    std::cout << std::string(total_width, '_') << std::endl;
    std::cout << std::setw(width) << "step" << " | ";
    
    for (long i = 0; i < num_names; i++) {
        std::cout << std::setw(width) << log_names[i];
        if (i < num_names - 1) {
            std::cout << " | ";
        }
    }

    std::cout << std::endl;
    std::cout << std::string(total_width, '_') << std::endl;
}


double Logger::get_value(const std::string& name) {
    std::string temp_name = name;
    size_t pos = name.find('/');
    if (pos != std::string::npos) {
        temp_name = name.substr(0, pos);
    }
    if (temp_name == "KE") {
        return particle.totalKineticEnergy();
    } else if (temp_name == "PE") {
        return particle.totalPotentialEnergy();
    } else if (temp_name == "TE") {
        return particle.totalEnergy();
    } else if (temp_name == "T") {
        return particle.calculateTemperature();
    } else if (temp_name == "phi") {
        return particle.getPackingFraction();
    } else {
        std::cerr << "Logger::get_value: Unknown name: " << temp_name << std::endl;
        exit(EXIT_FAILURE);
    }
}

double Logger::apply_modifier(std::string& name, double value) {
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/N") {
        value /= particle.n_particles;
        name = name.substr(0, name.size() - 2);
    } else if (name.size() >= 3 && name.substr(name.size() - 3) == "/Nv") {
        value /= particle.n_vertices;
        name = name.substr(0, name.size() - 3);
    }
    return value;
}

void Logger::write_values(long step, long width) {
    std::cout << std::setw(width) << step << " | " << std::setw(width);
    for (long i = 0; i < log_names.size(); i++) {
        double value = get_value(log_names[i]);
        apply_modifier(log_names[i], value);
        std::cout << std::setw(width) << apply_modifier(log_names[i], value);
        if (i < log_names.size() - 1) {
            std::cout << " | ";
        }
    }
    std::cout << std::endl;
}
#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/orchestrator.h"  
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

Logger::Logger(Particle& particle, const std::vector<std::string>& log_names, LoggerConfig config) : orchestrator(particle, log_names) {
    this->config = config;
}

Logger::~Logger() {
}

void Logger::write_header() {
    long num_names = orchestrator.log_names.size();
    long total_width = (config.width + 3) * (num_names + 1) - 1;
    std::cout << std::string(total_width, '_') << std::endl;
    std::cout << std::setw(config.width) << "step" << " | ";
    
    for (long i = 0; i < num_names; i++) {
        std::cout << std::setw(config.width) << orchestrator.log_names[i];
        if (i < num_names - 1) {
            std::cout << " | ";
        }
    }

    std::cout << std::endl;
    std::cout << std::string(total_width, '_') << std::endl;
}

void Logger::write_values(long step) {
    orchestrator.precalculate();
    if (config.last_header_log_step >= config.header_log_step_frequency) {
        write_header();
        config.last_header_log_step = 0;
    } else {
        config.last_header_log_step += 1;
    }
    std::cout << std::setw(config.width) << step << " | " << std::setw(config.width);
    for (long i = 0; i < orchestrator.log_names.size(); i++) {
        double value = orchestrator.get_value(orchestrator.log_names[i]);
        value = orchestrator.apply_modifier(orchestrator.log_names[i], value);
        std::cout << std::setw(config.width) << std::scientific << std::setprecision(config.precision) << value;
        if (i < orchestrator.log_names.size() - 1) {
            std::cout << " | ";
        }
    }
    std::cout << std::endl;
}
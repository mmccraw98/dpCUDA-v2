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
    long total_width = (width + 3) * (num_names + 2) - 1;
    std::cout << std::string(total_width, '_') << std::endl;
    std::cout << std::setw(width) << "step" << " | " << std::setw(width) << "time" << " | ";
    
    for (long i = 0; i < num_names; i++) {
        std::cout << std::setw(width) << log_names[i];
        if (i < num_names - 1) {
            std::cout << " | ";
        }
    }

    std::cout << std::endl;
    std::cout << std::string(total_width, '_') << std::endl;
}

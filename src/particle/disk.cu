// FUNCTION DECLARATIONS

#include "../../include/constants.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/functors.h"
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

Disk::Disk(long n_particles, long seed) {
    std::cout << "Disk::Disk" << std::endl;
    this->n_particles = n_particles;
    this->seed = seed;
    initDynamicVariables();

    d_test_array.resize(n_particles);
}

Disk::~Disk() {
    std::cout << "Disk::~Disk" << std::endl;
    clearDynamicVariables();

    d_test_array.clear();
}

double Disk::getAreaImpl() {
    std::cout << "Disk::getAreaImpl" << std::endl;
    return thrust::transform_reduce(d_radii.begin(), d_radii.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

double Disk::getOverlapFractionImpl() {
    std::cout << "Disk::getOverlapFractionImpl" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    return 0.0;
}

void Disk::scalePositionsImpl(double scale_factor) {
    std::cout << "Disk::scalePositionsImpl" << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    std::cout << "scale_factor: " << scale_factor << std::endl;
    thrust::transform(d_positions.begin(), d_positions.end(), d_positions.begin(), thrust::placeholders::_1 * scale_factor);
}

void Disk::updatePositionsImpl(double dt) {
    std::cout << "Disk::updatePositionsImpl" << std::endl;
}

void Disk::updateMomentaImpl(double dt) {
    std::cout << "Disk::updateMomentaImpl" << std::endl;
}

void Disk::calculateForcesImpl() {
    std::cout << "Disk::calculateForcesImpl" << std::endl;
}
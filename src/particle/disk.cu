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
    this->n_dof = n_particles * N_DIM;
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
    thrust::transform(d_positions.begin(), d_positions.end(), thrust::make_constant_iterator(scale_factor), d_positions.begin(), thrust::multiplies<double>());
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

void Disk::calculateKineticEnergyImpl() {
    std::cout << "Disk::calculateKineticEnergyImpl" << std::endl;
    thrust::transform(d_momenta.begin(), d_momenta.end(), d_kinetic_energy.begin(), Square());
}

void Disk::updateNeighborListImpl() {
    std::cout << "Disk::updateNeighborListImpl" << std::endl;
}
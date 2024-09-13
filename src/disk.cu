// FUNCTION DECLARATIONS

#include "../include/constants.h"
#include "../include/particle.h"
#include "../include/disk.h"
#include "../include/cuda_constants.cuh"
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

    d_positions.resize(n_particles * N_DIM);
    d_momenta.resize(n_particles * N_DIM);
    d_forces.resize(n_particles * N_DIM);
    d_radii.resize(n_particles);
    d_masses.resize(n_particles);
    d_potential_energy.resize(n_particles);
    d_kinetic_energy.resize(n_particles);
    d_last_positions.resize(n_particles * N_DIM);
    d_neighbor_list.resize(n_particles);

    d_test_array.resize(n_particles);
}

Disk::~Disk() {
    std::cout << "Disk::~Disk" << std::endl;
    d_positions.clear();
    d_momenta.clear();
    d_forces.clear();
    d_radii.clear();
    d_masses.clear();
    d_potential_energy.clear();
    d_kinetic_energy.clear();
    d_last_positions.clear();
    d_neighbor_list.clear();

    d_test_array.clear();
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
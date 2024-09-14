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

// Constructor
Disk::Disk(long n_particles, long seed) {
    std::cout << "Disk::Disk" << std::endl;
    this->n_particles = n_particles;
    this->seed = seed;
    this->n_dof = n_particles * N_DIM;
    initDynamicVariables();
    d_test_array.resize(n_particles);
}

// Destructor
Disk::~Disk() {
    std::cout << "Disk::~Disk" << std::endl;
    clearDynamicVariables();
    d_test_array.clear();
}

// Initialize dynamic variables
void Disk::initDynamicVariables() {
    std::cout << "Disk::initDynamicVariables" << std::endl;
    d_positions.resize(n_particles * N_DIM);
    d_last_positions.resize(n_particles * N_DIM);
    d_displacements.resize(n_particles * N_DIM);
    d_momenta.resize(n_particles * N_DIM);
    d_forces.resize(n_particles * N_DIM);
    d_radii.resize(n_particles);
    d_masses.resize(n_particles);
    d_potential_energy.resize(n_particles);
    d_kinetic_energy.resize(n_particles);
}

// Clear dynamic variables
void Disk::clearDynamicVariables() {
    std::cout << "Disk::clearDynamicVariables" << std::endl;
    d_positions.clear();
    d_last_positions.clear();
    d_displacements.clear();
    d_momenta.clear();
    d_forces.clear();
    d_radii.clear();
    d_masses.clear();
    d_potential_energy.clear();
    d_kinetic_energy.clear();
}

// Initialize geometric variables (if any)
void Disk::initGeometricVariables() {
    std::cout << "Disk::initGeometricVariables" << std::endl;
    // Add any necessary geometric variable initialization here if needed.
}

// Clear geometric variables (if any)
void Disk::clearGeometricVariables() {
    std::cout << "Disk::clearGeometricVariables" << std::endl;
    // Add any necessary geometric variable clearing here if needed.
}

// Set random positions in the simulation box
void Disk::setRandomPositions() {
    std::cout << "Disk::setRandomPositions" << std::endl;
    thrust::host_vector<double> box_size = getBoxSize();
    setRandomUniform(d_positions, 0.0, box_size[0]);
}

// Get area of the particles (disks in this case)
double Disk::getArea() const {
    std::cout << "Disk::getArea" << std::endl;
    return thrust::transform_reduce(d_radii.begin(), d_radii.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

// Get overlap fraction (currently a placeholder)
double Disk::getOverlapFraction() const {
    std::cout << "Disk::getOverlapFraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    return 0.0;
}

// Scale particle positions by a scale factor
void Disk::scalePositions(double scale_factor) {
    std::cout << "Disk::scalePositions" << std::endl;
    thrust::transform(d_positions.begin(), d_positions.end(), thrust::make_constant_iterator(scale_factor), d_positions.begin(), thrust::multiplies<double>());
}

// Update particle positions based on the time step (dt)
void Disk::updatePositions(double dt) {
    std::cout << "Disk::updatePositions" << std::endl;
    // Add logic to update positions based on time step.
}

// Update particle momenta based on the time step (dt)
void Disk::updateMomenta(double dt) {
    std::cout << "Disk::updateMomenta" << std::endl;
    // Add logic to update momenta based on time step.
}

// Calculate forces on the particles
void Disk::calculateForces() {
    std::cout << "Disk::calculateForces" << std::endl;
    // Add logic to calculate forces between particles.
}

// Calculate kinetic energy of the particles
void Disk::calculateKineticEnergy() {
    std::cout << "Disk::calculateKineticEnergy" << std::endl;
    thrust::transform(d_momenta.begin(), d_momenta.end(), d_kinetic_energy.begin(), Square());
}

// Update neighbor list for the particles (e.g., for force calculations)
void Disk::updateNeighborList() {
    std::cout << "Disk::updateNeighborList" << std::endl;
    // Add logic to update the neighbor list.
}

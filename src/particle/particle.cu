#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/dynamics.cuh"
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
Particle::Particle() {
}

// Destructor (virtual to allow proper cleanup in derived classes)
Particle::~Particle() {
}

// ----------------------------------------------------------------------
// ----------------------- Template Methods -----------------------------
// ----------------------------------------------------------------------

std::unordered_map<std::string, std::any> Particle::getArrayMap() {
    std::unordered_map<std::string, std::any> array_map;
    array_map["d_positions"]        = &d_positions;
    array_map["d_last_positions"]   = &d_last_positions;
    array_map["d_velocities"]       = &d_velocities;
    array_map["d_forces"]           = &d_forces;
    array_map["d_radii"]            = &d_radii;
    array_map["d_masses"]           = &d_masses;
    array_map["d_potential_energy"] = &d_potential_energy;
    array_map["d_kinetic_energy"]   = &d_kinetic_energy;
    array_map["d_neighbor_list"]    = &d_neighbor_list;
    return array_map;
}

// ----------------------------------------------------------------------
// -------------------- Universally Defined Methods ---------------------
// ----------------------------------------------------------------------

void Particle::setSeed(long seed) {
    if (seed == -1) {
        seed = time(0);
    }
    this->seed = seed;
    srand(seed);
}

void Particle::setKernelDimensions(long dim_block) {
    this->dim_block = dim_block;
    this->dim_grid = (n_particles + dim_block - 1) / dim_block;
    this->dim_vertex_grid = (n_vertices + dim_block - 1) / dim_block;
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_dim_block, &dim_block, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setKernelDimensions: Error copying dimBlock to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_grid, &dim_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setKernelDimensions: Error copying dimGrid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_vertex_grid, &dim_vertex_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setKernelDimensions: Error copying dimVertexGrid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::setNumParticles(long n_particles) {
    this->n_particles = n_particles;
    this->n_dof = n_particles * N_DIM;
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_n_particles, &n_particles, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setNumParticles: Error copying number of particles to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::setNumVertices(long n_vertices) {
    this->n_vertices = n_vertices;
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_n_vertices, &n_vertices, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setNumVertices: Error copying number of vertices to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::initDynamicVariables() {
    // Resize the device vectors
    d_positions.resize(n_particles * N_DIM);
    d_last_positions.resize(n_particles * N_DIM);
    d_displacements.resize(n_particles * N_DIM);
    d_velocities.resize(n_particles * N_DIM);
    d_forces.resize(n_particles * N_DIM);
    d_radii.resize(n_particles);
    d_masses.resize(n_particles);
    d_potential_energy.resize(n_particles);
    d_kinetic_energy.resize(n_particles);

    // Cast the raw pointers
    d_positions_ptr = thrust::raw_pointer_cast(&d_positions[0]);
    d_last_positions_ptr = thrust::raw_pointer_cast(&d_last_positions[0]);
    d_displacements_ptr = thrust::raw_pointer_cast(&d_displacements[0]);
    d_velocities_ptr = thrust::raw_pointer_cast(&d_velocities[0]);
    d_forces_ptr = thrust::raw_pointer_cast(&d_forces[0]);
    d_radii_ptr = thrust::raw_pointer_cast(&d_radii[0]);
    d_masses_ptr = thrust::raw_pointer_cast(&d_masses[0]);
    d_potential_energy_ptr = thrust::raw_pointer_cast(&d_potential_energy[0]);
    d_kinetic_energy_ptr = thrust::raw_pointer_cast(&d_kinetic_energy[0]);
}

void Particle::clearDynamicVariables() {
    // Clear the device vectors
    d_positions.clear();
    d_last_positions.clear();
    d_displacements.clear();
    d_velocities.clear();
    d_forces.clear();
    d_radii.clear();
    d_masses.clear();
    d_potential_energy.clear();
    d_kinetic_energy.clear();

    // Clear the pointers
    d_positions_ptr = nullptr;
    d_last_positions_ptr = nullptr;
    d_displacements_ptr = nullptr;
    d_velocities_ptr = nullptr;
    d_forces_ptr = nullptr;
    d_radii_ptr = nullptr;
    d_masses_ptr = nullptr;
    d_potential_energy_ptr = nullptr;
    d_kinetic_energy_ptr = nullptr;
}

void Particle::setBoxSize(const thrust::host_vector<double>& box_size) {
    if (box_size.size() != N_DIM) {
        throw std::invalid_argument("Particle::setBoxSize: Error box_size (" + std::to_string(box_size.size()) + ")" + " != " + std::to_string(N_DIM) + " elements");
    }
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_box_size, box_size.data(), sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setBoxSize: Error copying box size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

thrust::host_vector<double> Particle::getBoxSize() {
    thrust::host_vector<double> box_size(N_DIM);
    cudaError_t cuda_err = cudaMemcpyFromSymbol(&box_size[0], d_box_size, sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::getBoxSize: Error copying box size to host: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return box_size;
}

void Particle::initializeBox(double area) {
    double side_length = std::pow(area, 1.0 / N_DIM);
    thrust::host_vector<double> box_size(N_DIM, side_length);
    setBoxSize(box_size);
}

void Particle::setRandomUniform(thrust::device_vector<double>& values, double min, double max) {
    thrust::counting_iterator<long> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomUniform(min, max, seed));
}

void Particle::setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev) {
    thrust::counting_iterator<long> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomNormal(mean, stddev, seed));
}

void Particle::setRandomPositions() {
    thrust::host_vector<double> box_size = getBoxSize();
    setRandomUniform(d_positions, 0.0, box_size[0]);
}

double Particle::getDiameter(std::string which) {
    if (which == "min") {
        return 2.0 * *thrust::min_element(d_radii.begin(), d_radii.end());
    } else if (which == "max") {
        return 2.0 * *thrust::max_element(d_radii.begin(), d_radii.end());
    } else if (which == "mean") {
        return 2.0 * thrust::reduce(d_radii.begin(), d_radii.end()) / d_radii.size();
    } else {
        throw std::invalid_argument("Particle::getDiameter: which must be 'min', 'max', or 'mean', not " + which);
    }
}

void Particle::setBiDispersity(double size_ratio, double count_ratio) {
    if (size_ratio < 1.0) {
        throw std::invalid_argument("Particle::setBiDispersity: size_ratio must be > 1.0");
    }
    if (count_ratio < 0.0 || count_ratio > 1.0) {
        throw std::invalid_argument("Particle::setBiDispersity: count_ratio must be < 1.0 and > 0.0");
    }
    thrust::host_vector<double> radii(n_particles);
    long n_large = static_cast<long>(n_particles * count_ratio);
    double r_large = size_ratio;
    double r_small = 1.0;
    for (long i = 0; i < n_large; i++) {
        radii[i] = r_large / 2.0;
    }
    for (long i = n_large; i < n_particles; i++) {
        radii[i] = r_small / 2.0;
    }
    setArray("d_radii", radii);
}

double Particle::getBoxArea() {
    thrust::host_vector<double> box_size = getBoxSize();
    return thrust::reduce(box_size.begin(), box_size.end(), 1.0, thrust::multiplies<double>());
}

double Particle::getPackingFraction() {
    double box_area = getBoxArea();
    double area = getArea();
    return area / box_area;
}

double Particle::getDensity() {
    return getPackingFraction() - getOverlapFraction();
}

void Particle::scaleToPackingFraction(double packing_fraction) {
    double new_side_length = std::pow(getArea() / packing_fraction, 1.0 / N_DIM);
    double side_length = std::pow(getBoxArea(), 1.0 / N_DIM);
    scalePositions(new_side_length / side_length);
    setBoxSize(thrust::host_vector<double>(N_DIM, new_side_length));
}

double Particle::totalKineticEnergy() const {
    thrust::host_vector<double> h_kinetic_energy = d_kinetic_energy;
    return thrust::reduce(h_kinetic_energy.begin(), h_kinetic_energy.end(), 0.0, thrust::plus<double>());
}

double Particle::totalPotentialEnergy() const {
    return thrust::reduce(d_potential_energy.begin(), d_potential_energy.end(), 0.0, thrust::plus<double>());
}

double Particle::totalEnergy() const {
    return totalKineticEnergy() + totalPotentialEnergy();
}

void Particle::scalePositions(double scale_factor) {
    thrust::transform(d_positions.begin(), d_positions.end(), thrust::make_constant_iterator(scale_factor), d_positions.begin(), thrust::multiplies<double>());
}

void Particle::updatePositions(double dt) {
    kernelUpdatePositions<<<dim_grid, dim_block>>>(d_positions_ptr, d_velocities_ptr, dt);
}

void Particle::updateVelocities(double dt) {
    kernelUpdateVelocities<<<dim_grid, dim_block>>>(d_velocities_ptr, d_forces_ptr, d_masses_ptr, dt);
}

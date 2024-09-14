#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
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

// Method to create a map of device arrays
std::unordered_map<std::string, std::any> Particle::getArrayMap() {
    std::unordered_map<std::string, std::any> array_map;
    array_map["d_positions"]        = &d_positions;
    array_map["d_last_positions"]   = &d_last_positions;
    array_map["d_momenta"]          = &d_momenta;
    array_map["d_forces"]           = &d_forces;
    array_map["d_radii"]            = &d_radii;
    array_map["d_masses"]           = &d_masses;
    array_map["d_potential_energy"] = &d_potential_energy;
    array_map["d_kinetic_energy"]   = &d_kinetic_energy;
    array_map["d_neighbor_list"]    = &d_neighbor_list;
    return array_map;
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
    thrust::transform(values.begin(), values.end(), values.begin(), RandomUniform(min, max, seed));
}

void Particle::setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev) {
    thrust::transform(values.begin(), values.end(), values.begin(), RandomNormal(mean, stddev, seed));
}

// Implemented Functions
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

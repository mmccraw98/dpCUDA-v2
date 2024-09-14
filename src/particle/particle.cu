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

// Forward declaration derived classes
class Disk;

// Explicit instantiation of the derived classes
template class Particle<Disk>;

template <typename Derived>
Particle<Derived>::Particle() {
}

template <typename Derived>
Particle<Derived>::~Particle() {
}

// Method to create a map of device arrays
template <typename Derived>
std::unordered_map<std::string, std::any> Particle<Derived>::getArrayMap() {
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

// Method to retrieve a device array by name and return it as a host vector
template <typename Derived>
template <typename T>
thrust::host_vector<T> Particle<Derived>::getArray(const std::string& array_name) {
    auto array_map = getArrayMap();
    auto it = array_map.find(array_name);
    if (it != array_map.end()) {
        if (it->second.type() == typeid(thrust::device_vector<T>*)) {
            auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
            thrust::host_vector<T> host_array(vec_ptr->size());
            thrust::copy(vec_ptr->begin(), vec_ptr->end(), host_array.begin());
            return host_array;
        } else {
            throw std::runtime_error("Type mismatch for array: " + array_name);
        }
    } else {
        throw std::runtime_error("Array not found: " + array_name);
    }
}

// Method to set a device array by name from a host vector
template <typename Derived>
template <typename T>
void Particle<Derived>::setArray(const std::string& array_name, const thrust::host_vector<T>& host_array) {
    auto array_map = getArrayMap();
    auto it = array_map.find(array_name);
    if (it != array_map.end()) {
        if (it->second.type() == typeid(thrust::device_vector<T>*)) {
            auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
            if (host_array.size() != vec_ptr->size()) {
                throw std::out_of_range("Size mismatch between host and device arrays for: " + array_name);
            }
            thrust::copy(host_array.begin(), host_array.end(), vec_ptr->begin());
        } else {
            throw std::runtime_error("Type mismatch for array: " + array_name);
        }
    } else {
        throw std::runtime_error("Array not found: " + array_name);
    }
}

template <typename Derived>
void Particle<Derived>::setBoxSize(const thrust::host_vector<double>& box_size) {
    if (box_size.size() != N_DIM) {
        throw std::invalid_argument("Particle::setBoxSize: Error box_size (" + std::to_string(box_size.size()) + ")" + " != " + std::to_string(N_DIM) + " elements");
    }
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_box_size, box_size.data(), sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setBoxSize: Error copying box size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename Derived>
thrust::host_vector<double> Particle<Derived>::getBoxSize() {
    thrust::host_vector<double> box_size(N_DIM);
    cudaError_t cuda_err = cudaMemcpyFromSymbol(&box_size[0], d_box_size, sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::getBoxSize: Error copying box size to host: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return box_size;
}

template <typename Derived>
void Particle<Derived>::initializeBox(double area) {
    double side_length = std::pow(area, 1.0 / N_DIM);
    thrust::host_vector<double> box_size(N_DIM, side_length);
    setBoxSize(box_size);
}

template <typename Derived>
void Particle<Derived>::setRandomUniform(thrust::device_vector<double>& values, double min, double max) {
    thrust::transform(values.begin(), values.end(), values.begin(), RandomUniform(min, max, seed));
}

template <typename Derived>
void Particle<Derived>::setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev) {
    thrust::transform(values.begin(), values.end(), values.begin(), RandomNormal(mean, stddev, seed));
}

template <typename Derived>
void Particle<Derived>::initDynamicVariables() {
    std::cout << "Particle<Derived>::initDynamicVariables" << std::endl;
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

template <typename Derived>
void Particle<Derived>::clearDynamicVariables() {
    std::cout << "Particle<Derived>::clearDynamicVariables" << std::endl;
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

template <typename Derived>
void Particle<Derived>::setRandomPositions() {
    thrust::host_vector<double> box_size = getBoxSize();
    setRandomUniform(d_positions, 0.0, box_size[0]);
}

template <typename Derived>
double Particle<Derived>::getDiameter(std::string which) {
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

template <typename Derived>
void Particle<Derived>::setBiDispersity(double size_ratio, double count_ratio) {
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

template <typename Derived>
double Particle<Derived>::getBoxArea() {
    thrust::host_vector<double> box_size = getBoxSize();
    return thrust::reduce(box_size.begin(), box_size.end(), 1.0, thrust::multiplies<double>());
}

template <typename Derived>
double Particle<Derived>::getPackingFraction() {
    double box_area = getBoxArea();
    double area = getArea();
    return area / box_area;
}

template <typename Derived>
double Particle<Derived>::getDensity() {
    return getPackingFraction() - getOverlapFraction();
}

template <typename Derived>
void Particle<Derived>::scaleToPackingFraction(double packing_fraction) {
    double new_side_length = std::pow(getArea() / packing_fraction, 1.0 / N_DIM);
    double side_length = std::pow(getBoxArea(), 1.0 / N_DIM);
    scalePositions(new_side_length / side_length);
    setBoxSize(thrust::host_vector<double>(N_DIM, new_side_length));
}

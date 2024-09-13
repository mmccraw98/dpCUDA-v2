// src/particles/Particle.cu

#include <iostream>
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <thrust/device_vector.h>
#include "particles/Particle.cuh"

template <typename Derived>
std::unordered_map<std::string, std::any> Particle<Derived>::getArrayMap() {
    std::unordered_map<std::string, std::any> array_map;

    // Double arrays
    array_map["d_positions"]        = &d_positions;
    array_map["d_momenta"]          = &d_momenta;
    array_map["d_forces"]           = &d_forces;
    array_map["d_radii"]            = &d_radii;
    array_map["d_masses"]           = &d_masses;
    array_map["d_potential_energy"] = &d_potential_energy;
    array_map["d_kinetic_energy"]   = &d_kinetic_energy;
    array_map["d_last_positions"]   = &d_last_positions;
    array_map["d_box_size"]         = &d_box_size;

    // Long arrays
    array_map["d_neighbor_list"]    = &d_neighbor_list;

    return array_map;
}

template <typename Derived>
template <typename T>
T Particle<Derived>::getArrayValue(const std::string& array_name, size_t index) {
    auto array_map = static_cast<Derived*>(this)->getArrayMap();
    auto it = array_map.find(array_name);

    if (it != array_map.end()) {
        if (it->second.type() == typeid(thrust::device_vector<T>*)) {
            auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
            if (index >= vec_ptr->size()) {
                throw std::out_of_range("Index out of range for array: " + array_name);
            }
            T value;
            thrust::copy(vec_ptr->begin() + index, vec_ptr->begin() + index + 1, &value);
            return value;
        } else {
            throw std::runtime_error("Type mismatch for array: " + array_name);
        }
    } else {
        throw std::runtime_error("Array not found: " + array_name);
    }
}

template <typename Derived>
template <typename T>
void Particle<Derived>::setArrayValue(const std::string& array_name, size_t index, const T& value) {
    auto array_map = static_cast<Derived*>(this)->getArrayMap();
    auto it = array_map.find(array_name);

    if (it != array_map.end()) {
        if (it->second.type() == typeid(thrust::device_vector<T>*)) {
            auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
            if (index >= vec_ptr->size()) {
                throw std::out_of_range("Index out of range for array: " + array_name);
            }
            thrust::copy(&value, &value + 1, vec_ptr->begin() + index);
        } else {
            throw std::runtime_error("Type mismatch for array: " + array_name);
        }
    } else {
        throw std::runtime_error("Array not found: " + array_name);
    }
}
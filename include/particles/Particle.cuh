// include/particles/Particle.cuh
#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "Constants.h"
#include "../include/kernels/CudaConstants.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <unordered_map>
#include <cmath>

// design standards:
// keep everything as flexible as possible
// keep everything as performance oriented as possible
// keep everything as straightforward and simple as possible (isolate functionality as much as possible)

// different particle types will have different updates to their dynamic variables (translation, rotation, etc.)
// different integrators will use different combinations / orders of these updates

template <typename Derived>
class Particle {
public:

    thrust::device_vector<double> d_positions;  // generalized position
    thrust::device_vector<double> d_last_positions;  // for tracking displacement
    thrust::device_vector<double> d_displacements;  // for storing displacement: positions - last_positions
    thrust::device_vector<double> d_momenta;  // generalized momentum
    thrust::device_vector<double> d_forces;  // generalized force
    thrust::device_vector<double> d_radii;  // particle radii
    thrust::device_vector<double> d_masses;  // generalized masses
    thrust::device_vector<double> d_potential_energy;
    thrust::device_vector<double> d_kinetic_energy;
    thrust::device_vector<long> d_neighbor_list;
    thrust::device_vector<double> d_box_size;
    double e_c;
    double neighbor_cutoff;
    long max_neighbors;
    long n_particles;
    long n_dim = N_DIM;
    long n_dof;

    // --------------------- Utility Methods ---------------------

    /**
     * @brief Creates a key-value map of pointers to the device arrays.
     * 
     * @return std::unordered_map<std::string, std::any> 
     */
    std::unordered_map<std::string, std::any> getArrayMap() {
        std::unordered_map<std::string, std::any> array_map;

        // Double arrays
        array_map["d_positions"]        = &d_positions;
        array_map["d_last_positions"]   = &d_last_positions;
        array_map["d_momenta"]          = &d_momenta;
        array_map["d_forces"]           = &d_forces;
        array_map["d_radii"]            = &d_radii;
        array_map["d_masses"]           = &d_masses;
        array_map["d_potential_energy"] = &d_potential_energy;
        array_map["d_kinetic_energy"]   = &d_kinetic_energy;
        array_map["d_box_size"]         = &d_box_size;

        // Long arrays
        array_map["d_neighbor_list"]    = &d_neighbor_list;

        return array_map;
    }

    /**
     * @brief Retrieves the device array by name as a host vector.
     * 
     * @tparam T 
     * @param array_name name of the array to retrieve
     * @return thrust::host_vector<T> 
     */
    template <typename T>
    thrust::host_vector<T> getArray(const std::string& array_name) {
        auto array_map = getArrayMap();
        auto it = array_map.find(array_name);

        if (it != array_map.end()) {
            if (it->second.type() == typeid(thrust::device_vector<T>*)) {
                auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
                // Create a host vector and copy device data to host
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

    /**
     * @brief Sets the device array by name from a host vector.
     * 
     * @tparam T 
     * @param array_name name of the array to set
     * @param host_array host vector to set the array to
     */
    template <typename T>
    void setArray(const std::string& array_name, const thrust::host_vector<T>& host_array) {
        auto array_map = getArrayMap();
        auto it = array_map.find(array_name);

        if (it != array_map.end()) {
            if (it->second.type() == typeid(thrust::device_vector<T>*)) {
                auto vec_ptr = std::any_cast<thrust::device_vector<T>*>(it->second);
                if (host_array.size() != vec_ptr->size()) {
                    throw std::out_of_range("Size mismatch between host and device arrays for: " + array_name);
                }
                // Copy host data back to device
                thrust::copy(host_array.begin(), host_array.end(), vec_ptr->begin());
            } else {
                throw std::runtime_error("Type mismatch for array: " + array_name);
            }
        } else {
            throw std::runtime_error("Array not found: " + array_name);
        }
    }
    // ------------------- Simulation Methods --------------------

    void setBoxSize(thrust::host_vector<double> &box_size) {
        std::cout << "Particle::setBoxSize" << std::endl;
        d_box_size = box_size;
        double *box_size_ptr = thrust::raw_pointer_cast(&(d_box_size[0]));
        cudaMemcpyToSymbol(d_box_size_ptr, &box_size_ptr, sizeof(box_size_ptr));
    }

    thrust::host_vector<double> getBoxSize() {
        std::cout << "Particle::getBoxSize" << std::endl;
        thrust::host_vector<double> box_size;
        cudaMemcpyFromSymbol(&d_box_size, d_box_size_ptr, sizeof(d_box_size_ptr));
        box_size = d_box_size;
        return box_size;
    }

    /**
     * @brief Initializes the simulation box dimensions in d_box_size given an area.  Works for a general n-dimensional box.
     * 
     * @param area area of the box
     */
    void initializeBox(double area) {
        std::cout << "Particle::initializeBox" << std::endl;
        cudaMemcpyToSymbol(d_n_dim, &n_dim, sizeof(n_dim));
        d_box_size.resize(n_dim);
        double side_length = std::pow(area, 1.0 / n_dim);
        for (long dim = 0; dim < n_dim; dim++) {
            d_box_size[dim] = side_length;
        }
        setBoxSize(d_box_size);
    }

    void updatePositions(double dt) {
        std::cout << "Particle::updatePositions" << std::endl;
        static_cast<Derived*>(this)->updatePositionsImpl(dt);
    }

    void updateMomenta(double dt) {
        std::cout << "Particle::updateMomenta" << std::endl;
        static_cast<Derived*>(this)->updateMomentaImpl(dt);
    }

    void calculateForces() {
        std::cout << "Particle::calculateForces" << std::endl;
        static_cast<Derived*>(this)->calculateForcesImpl();
    }

    double totalKineticEnergy() const {
        std::cout << "Particle::totalKineticEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalKineticEnergyImpl();
    }

    double totalPotentialEnergy() const {
        std::cout << "Particle::totalPotentialEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalPotentialEnergyImpl();
    }

    double totalEnergy() const {
        std::cout << "Particle::totalEnergy" << std::endl;
        return static_cast<const Derived*>(this)->totalEnergyImpl();
    }
};

#endif // PARTICLE_CUH
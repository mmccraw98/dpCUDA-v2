#ifndef PARTICLE_H
#define PARTICLE_H

#include "../constants.h"
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Particle {
public:
    Particle();
    virtual ~Particle();  // Ensure virtual destructor for proper cleanup in derived classes

    // Device vectors for particle data
    thrust::device_vector<double> d_positions;
    thrust::device_vector<double> d_last_positions;
    thrust::device_vector<double> d_displacements;
    thrust::device_vector<double> d_momenta;
    thrust::device_vector<double> d_forces;
    thrust::device_vector<double> d_radii;
    thrust::device_vector<double> d_masses;
    thrust::device_vector<double> d_potential_energy;
    thrust::device_vector<double> d_kinetic_energy;
    thrust::device_vector<long> d_neighbor_list;

    // Simulation parameters
    double e_c;
    double neighbor_cutoff;
    long max_neighbors;
    long n_particles;
    long n_dof;
    long seed;

    // Universal Methods
    std::unordered_map<std::string, std::any> getArrayMap();

    template <typename T>
    thrust::host_vector<T> getArray(const std::string& array_name) {
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
                thrust::copy(host_array.begin(), host_array.end(), vec_ptr->begin());
            } else {
                throw std::runtime_error("Type mismatch for array: " + array_name);
            }
        } else {
            throw std::runtime_error("Array not found: " + array_name);
        }
    }
    
    void setBoxSize(const thrust::host_vector<double>& box_size);
    thrust::host_vector<double> getBoxSize();
    void initializeBox(double area);
    void setRandomUniform(thrust::device_vector<double>& values, double min, double max);
    void setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev);

    // Pure Virtual Functions (must be implemented in derived classes)
    virtual void initDynamicVariables() = 0;
    virtual void clearDynamicVariables() = 0;
    virtual void initGeometricVariables() = 0;
    virtual void clearGeometricVariables() = 0;
    virtual void setRandomPositions() = 0;
    virtual double getArea() const = 0;  // Derived must implement
    virtual double getOverlapFraction() const = 0;  // Derived must implement
    virtual void scalePositions(double scale_factor) = 0;
    virtual void updatePositions(double dt) = 0;
    virtual void updateMomenta(double dt) = 0;
    virtual void calculateForces() = 0;
    virtual void calculateKineticEnergy() = 0;
    virtual void updateNeighborList() = 0;

    // Methods with Implemented Logic (will use virtual calls if overridden)
    double getDiameter(std::string which = "min");
    void setBiDispersity(double size_ratio, double count_ratio);
    double getBoxArea();
    double getPackingFraction();
    double getDensity();
    void scaleToPackingFraction(double packing_fraction);

    double totalKineticEnergy() const;
    double totalPotentialEnergy() const;
    double totalEnergy() const;
};

#endif /* PARTICLE_H */
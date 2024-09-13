#ifndef PARTICLE_H
#define PARTICLE_H

#include "constants.h"
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// design standards:
// keep everything as flexible as possible
// keep everything as performance oriented as possible
// keep everything as straightforward and simple as possible (isolate functionality as much as possible)

// different particle types will have different updates to their dynamic variables (translation, rotation, etc.)
// different integrators will use different combinations / orders of these updates

template <typename Derived>
class Particle {
public:
    Particle();
    ~Particle();
    
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

    // Utility methods
    std::unordered_map<std::string, std::any> getArrayMap();
    
    template <typename T>
    thrust::host_vector<T> getArray(const std::string& array_name);
    
    template <typename T>
    void setArray(const std::string& array_name, const thrust::host_vector<T>& host_array);
    
    void initializeBox(double area);

    thrust::host_vector<double> getBoxSize();

    // Simulation methods using CRTP
    void updatePositions(double dt);
    void updateMomenta(double dt);
    void calculateForces();
    
    double totalKineticEnergy() const;
    double totalPotentialEnergy() const;
    double totalEnergy() const;
};

#endif /* PARTICLE_H */
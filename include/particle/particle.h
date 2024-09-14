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

// design standards:
// keep everything as flexible as possible
// keep everything as performance oriented as possible
// keep everything as straightforward and simple as possible (isolate functionality as much as possible)

// different particle types will have different updates to their dynamic variables (translation, rotation, etc.)
// different integrators will use different combinations / orders of these updates

// vertices are internal variables
// the common variables are the particle-level variables

// heirarchy of particles:
// particle
// |_ disk
// |_ ellipsoid (rotational degrees of freedom)
// |__ ga model and dimer (rotational with internal variables, but no internal dynamics)
// |___ smooth variants
// |_ dpm (internal variables with dynamics)
// |___ smooth variants

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
    long seed;

    // Universal Methods
    /**
     * @brief Get a key-value map for the pointers to the member device arrays (primarily used for the get/setArray methods)
     * 
     * @return std::unordered_map<std::string, std::any> 
     */
    std::unordered_map<std::string, std::any> getArrayMap();
    
    /**
     * @brief Get a member device array as a host vector
     * 
     * @param array_name name of the member device array to get
     * @return thrust::host_vector<T> 
     */
    template <typename T>
    thrust::host_vector<T> getArray(const std::string& array_name);
    
    /**
     * @brief Set a member device array from a host vector
     * 
     * @param array_name name of the member device array to set
     * @param host_array host vector to set the array from
     */
    template <typename T>
    void setArray(const std::string& array_name, const thrust::host_vector<T>& host_array);
    
    /**
     * @brief Set the box size from a host vector
     * 
     * @param box_size host vector of length N_DIM
     */
    void setBoxSize(const thrust::host_vector<double>& box_size);

    /**
     * @brief Get the box size as a host vector
     * 
     * @return thrust::host_vector<double> 
     */
    thrust::host_vector<double> getBoxSize();

    /**
     * @brief Set the box size as an N_DIM hypercube with side length derived from the generalized area
     * 
     * @param area generalized area of the simulation box (2d-area, 3d-volume)
     */
    void initializeBox(double area);

    /**
     * @brief Assign random uniform values to a device vector within a given range
     * 
     * @param values device vector to assign the random uniform values to
     * @param min minimum value
     * @param max maximum value
     */
    void setRandomUniform(thrust::device_vector<double>& values, double min, double max);

    /**
     * @brief Assign random normal values to a device vector within a given range
     * 
     * @param values device vector to assign the random normal values to
     * @param mean mean of the normal distribution
     * @param stddev standard deviation of the normal distribution
     */
    void setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev);

    // CRTP-Specific Methods
    void updatePositions(double dt);
    void updateMomenta(double dt);
    void calculateForces();
    
    double totalKineticEnergy() const;
    double totalPotentialEnergy() const;
    double totalEnergy() const;
};

#endif /* PARTICLE_H */
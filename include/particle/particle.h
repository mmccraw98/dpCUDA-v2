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

    /**
     * @brief Initialize the dynamic variables (positions, momenta, etc. whatever is generally needed for the simulation)
     * 
     */
    void initDynamicVariables();

    /**
     * @brief Clear the dynamic variables (positions, momenta, etc. whatever is generally needed for the simulation)
     * 
     */
    void clearDynamicVariables();

    /**
     * @brief Initialize the geometric variables if relevant (areas, angles, lengths, etc.)
     * 
     */
    void initGeometricVariables();

    /**
     * @brief Clear the geometric variables if relevant (areas, angles, lengths, etc.)
     * 
     */
    void clearGeometricVariables();

    /**
     * @brief Uniformly distribute the particle positions in the simulation box
     * 
     */
    void setRandomPositions();

    /**
     * @brief Get the diameter of the particles
     * 
     * @param which which diameter to get ("min", "max", or "mean")
     * @return double 
     */
    double getDiameter(std::string which = "min");

    /**
     * @brief Set the bi-dispersity of the particles given a size ratio (large/small diameter) and a count ratio (large/small number)
     * 
     * @param size_ratio ratio of the large diameter to the small diameter
     * @param count_ratio ratio of the large number to the small number
     */
    void setBiDispersity(double size_ratio, double count_ratio);

    /**
     * @brief Get the area of the simulation box
     * 
     * @return double 
     */
    double getBoxArea();

    /**
     * @brief Get the area of the particles
     * 
     */
    double getArea() {
        return static_cast<Derived*>(this)->getAreaImpl();
    }

    /**
     * @brief Get the packing fraction of the system
     * 
     * @return double 
     */
    double getPackingFraction();

    /**
     * @brief Get the overlap fraction of the system
     * 
     * @return double 
     */
    double getOverlapFraction() {
        return static_cast<Derived*>(this)->getOverlapFractionImpl();
    }

    /**
     * @brief Get the density of the system: packing fraction - overlap fraction
     * 
     * @return double 
     */
    double getDensity();

    /**
     * @brief Scale the positions of the particles by a given factor
     * 
     * @param scale_factor 
     */
    void scalePositions(double scale_factor) {
        static_cast<Derived*>(this)->scalePositionsImpl(scale_factor);
    }

    /**
     * @brief Scale the system size so that the particles are at a given packing fraction
     * 
     * @param packing_fraction packing fraction to scale to
     */
    void scaleToPackingFraction(double packing_fraction);

    void updatePositions(double dt) {
        static_cast<Derived*>(this)->updatePositionsImpl(dt);
    }
    void updateMomenta(double dt) {
        static_cast<Derived*>(this)->updateMomentaImpl(dt);
    }
    void calculateForces() {
        static_cast<Derived*>(this)->calculateForcesImpl();
    }
    void calculateKineticEnergy() {
        static_cast<Derived*>(this)->calculateKineticEnergyImpl();
    }
    void calculatePotentialEnergy() {
        static_cast<Derived*>(this)->calculatePotentialEnergyImpl();
    }


    inline double totalKineticEnergy() const {
        return thrust::reduce(d_kinetic_energy.begin(), d_kinetic_energy.end(), 0.0, thrust::plus<double>());
    }
    inline double totalPotentialEnergy() const {
        return thrust::reduce(d_potential_energy.begin(), d_potential_energy.end(), 0.0, thrust::plus<double>());
    }
    inline double totalEnergy() const {
        return totalKineticEnergy() + totalPotentialEnergy();
    }
};

#endif /* PARTICLE_H */
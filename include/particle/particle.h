#ifndef PARTICLE_H
#define PARTICLE_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/kernels/kernels.cuh"
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
    thrust::device_vector<double> d_positions;  // particle positions
    thrust::device_vector<double> d_last_positions;  // particle positions at the last neighbor list update
    thrust::device_vector<double> d_displacements;  // displacement between current and last positions
    thrust::device_vector<double> d_velocities;  // particle velocities
    thrust::device_vector<double> d_forces;  // forces on the particles
    thrust::device_vector<double> d_radii;  // particle radii
    thrust::device_vector<double> d_masses;  // particle masses
    thrust::device_vector<double> d_potential_energy;  // potential energy of the particles
    thrust::device_vector<double> d_kinetic_energy;  // kinetic energy of the particles
    thrust::device_vector<long> d_neighbor_list;  // neighbor list for the particles
    thrust::device_vector<long> d_num_neighbors;  // number of neighbors for each particle

    // Pointers to the device arrays
    double* d_positions_ptr;
    double* d_last_positions_ptr;
    double* d_displacements_ptr;
    double* d_velocities_ptr;
    double* d_forces_ptr;
    double* d_radii_ptr;
    double* d_masses_ptr;
    double* d_potential_energy_ptr;
    double* d_kinetic_energy_ptr;

    // Simulation parameters
    double e_c = -1, e_a = -1, e_b = -1, e_l = -1;  // energy scales for interaction, area, bending, and length
    double n_c = -1, n_a = -1, n_b = -1, n_l = -1;  // exponents for the energy terms
    double neighbor_cutoff;  // cutoff distance for the neighbor list
    long max_neighbors;  // maximum number of neighbors
    long max_neighbors_allocated;  // maximum number of neighbors allocated for each particle
    long n_particles = -1;  // total number of particles
    long n_vertices = -1;  // total number of vertices
    long n_dof;  // number of degrees of freedom
    long seed;  // random number generator seed
    long dim_grid, dim_block, dim_vertex_grid;  // dimensions for the CUDA kernels

    // ----------------------------------------------------------------------
    // ----------------------- Template Methods -----------------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Get a key-value map to pointers for all the member device arrays.
     * Serves as a utility function for the getArray and setArray methods to reduce
     * code duplication.
     * 
     * @return The array map.
     */
    std::unordered_map<std::string, std::any> getArrayMap();

    /**
     * @brief Get a host array from a device array.
     * 
     * @tparam T The type of the array.
     * @param array_name The name of the array.
     * @return The host array.
     */
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

    /**
     * @brief Set a device array from a host array.
     * 
     * @tparam T The type of the array.
     * @param array_name The name of the array.
     * @param host_array The host array.
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
                thrust::copy(host_array.begin(), host_array.end(), vec_ptr->begin());
            } else {
                throw std::runtime_error("Type mismatch for array: " + array_name);
            }
        } else {
            throw std::runtime_error("Array not found: " + array_name);
        }
    }

    // ----------------------------------------------------------------------
    // -------------------- Universally Defined Methods ---------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Set the seed for the random number generator.
     * 
     * @param seed The seed for the random number generator.
     */
    void setSeed(long seed);

    /**
     * @brief Set the dimensions for the CUDA kernels.
     * 
     * @param dim_block The number of threads in the block (default is 256).
     */
    void setKernelDimensions(long dim_block = 256);

    /**
     * @brief Virtual method to set the number of particles.
     * 
     * @param n_particles The number of particles.
     */
    virtual void setNumParticles(long n_particles);

    /**
     * @brief Set the number of vertices for the particles.
     * 
     * @param n_vertices The number of vertices.
     */
    virtual void setNumVertices(long n_vertices);

    /**
     * @brief Initialize the dynamic variables and their pointers given the number of particles and dimensions.
     */
    virtual void initDynamicVariables();

    /**
     * @brief Clear the dynamic variables and their pointers from the device memory.
     */
    virtual void clearDynamicVariables();

    /**
     * @brief Set the box size vector constant in the device memory.
     * 
     * @param box_size The box size vector.
     */
    void setBoxSize(const thrust::host_vector<double>& box_size);

    /**
     * @brief Get the box size vector from the device memory.
     * 
     * @return The box size vector in the host memory.
     */
    thrust::host_vector<double> getBoxSize();

    /**
     * @brief Synchronize the neighbor list on the device.
     */
    virtual void syncNeighborList();
    
    /**
     * @brief Set the energy scales for the particles.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param e The energy scale.
     * @param which The type of energy scale to set ("c", "a", "b", or "l").
     */
    void setEnergyScale(double e, std::string which);

    /**
     * @brief Set the exponents for the energy terms.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param n The exponent.
     * @param which The type of exponent to set ("c", "a", "b", or "l").
     */
    void setExponent(double n, std::string which);


    virtual void setCudaConstants();

    virtual void getCudaConstants();

    /**
     * @brief Initialize the box size given the desired area of the box.
     * 
     * @param area The area of the box (default is 1.0).
     */
    void initializeBox(double area = 1.0);

    /**
     * @brief Fill a device vector with random uniform values.
     * 
     * @param values The device vector.
     * @param min The minimum value.
     * @param max The maximum value.
     */
    void setRandomUniform(thrust::device_vector<double>& values, double min, double max);

    /**
     * @brief Fill a device vector with random normal values.
     * 
     * @param values The device vector.
     * @param mean The mean value.
     * @param stddev The standard deviation.
     */
    void setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev);

    /**
     * @brief Set the particle positions to random uniform values within the box size.
     */
    virtual void setRandomPositions();

    /**
     * @brief Get the diameter of the particles.
     * 
     * @param which The type of diameter to get ("min" for minimum, "max" for maximum, or "avg" for average).
     * @return The diameter of the particles.
     */
    virtual double getDiameter(std::string which = "min");

    /**
     * @brief Set the bi-dispersity of the particles given the size ratio (large/small) and the count ratio (large/small).
     * 
     * @param size_ratio The ratio of the maximum to minimum diameter (must be > 1.0).
     * @param count_ratio The ratio of the number of large particles to the number of small particles (must be < 1.0 and > 0.0).
     */
    void setBiDispersity(double size_ratio, double count_ratio);

    /**
     * @brief Get the area of the box.
     * 
     * @return The area of the box.
     */
    double getBoxArea();

    /**
     * @brief Get the packing fraction of the particles.
     * 
     * @return The packing fraction of the particles.
     */
    double getPackingFraction();
    
    /**
     * @brief Get the density of the particles as packing fraction - overlap fraction.
     * 
     * @return The density of the particles.
     */
    double getDensity();
    
    /**
     * @brief Scale the box size and the particle positions to set a desired packing fraction.
     * Assumes that the particle diameters are already set.
     * 
     * @param packing_fraction The new packing fraction.
     */
    virtual void scaleToPackingFraction(double packing_fraction);

    /**
     * @brief Get the total kinetic energy of the particles.
     * 
     * @return The total kinetic energy.
     */
    double totalKineticEnergy() const;

    /**
     * @brief Get the total potential energy of the particles.
     * 
     * @return The total potential energy.
     */
    double totalPotentialEnergy() const;

    /**
     * @brief Get the total energy of the particles.
     * 
     * @return The total energy.
     */
    double totalEnergy() const;

    /**
     * @brief Initialize the geometric variables and their pointers for the object - default is empty.
     */
    virtual void initGeometricVariables() {};

    /**
     * @brief Clear the geometric variables and their pointers for the object - default is empty.
     */
    virtual void clearGeometricVariables() {};

    /**
     * @brief Apply an affine transformation to the positions of the particles.
     * 
     * @param scale_factor The scaling factor.
     */
    virtual void scalePositions(double scale_factor);

    /**
     * @brief Update the positions of the particles using an explicit Euler method.
     * x(t+dt) = x(t) + dt * v(t)
     * 
     * @param dt The time step.
     */
    virtual void updatePositions(double dt);

    /**
     * @brief Update the velocities of the particles using an explicit Euler method.
     * 
     * @param dt The time step.
     */
    virtual void updateVelocities(double dt);

    /**
     * @brief Get the maximum displacement of the particles since the last neighbor list update.
     * 
     * @return The maximum displacement.
     */
    virtual double getMaxDisplacement();

    /**
     * @brief Check if the neighbor list of the particles needs to be updated (if maximum displacement is greater than the neighbor cutoff).
     */
    virtual void checkForNeighborUpdate();

    /**
     * @brief Calculate the neighbor list for the particles.
     */
    virtual void updateNeighborList();

    // ----------------------------------------------------------------------
    // ---------------------- Pure Virtual Methods --------------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Virtual method to get the total area of the particles.
     * 
     * @return The total area of the particles.
     */
    virtual double getArea() const = 0;

    /**
     * @brief Virtual method to calculate the ratio of the particle overlap area to the total area.
     * 
     * @return The overlap fraction of the particles.
     */
    virtual double getOverlapFraction() const = 0;

    /**
     * @brief Virtual method to calculate the forces on the particles.
     */
    virtual void calculateForces() = 0;

    /**
     * @brief Virtual method to calculate the kinetic energy of the particles.
     */
    virtual void calculateKineticEnergy() = 0;
};

#endif /* PARTICLE_H */
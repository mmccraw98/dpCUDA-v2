#ifndef PARTICLE_H
#define PARTICLE_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "config.h"

#include <unordered_map>
#include <any>
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename Derived>
class Particle {
public:
    Particle();
    ~Particle();

    template <typename ConfigType>
    void initialize(const ConfigType& config) {
        static_assert(std::is_base_of<BaseParticleConfig, ConfigType>::value, 
                      "ConfigType must derive from BaseParticleConfig");

        // Initialize common fields
        this->type_name = config.type_name;

        // Delegate to the specific particle class (Derived) for further initialization
        static_cast<Derived*>(this)->initializeFromConfig(config);
    }

    // These arrays (and the parameters) have to be saved to be able to restart from a configuration - all other values can be derived if not defined
    std::vector<std::string> fundamental_values = {"d_positions", "d_velocities"};
    // These are the values that need to be calculated before the log value is calculated
    std::vector<std::string> pre_req_calculations = {"KE", "T"};

    std::string type_name;

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
    double neighbor_cutoff = -1;  // cutoff distance for the neighbor list
    long max_neighbors = -1;  // maximum number of neighbors
    long max_neighbors_allocated = -1;  // maximum number of neighbors allocated for each particle
    long n_particles = -1;  // total number of particles
    long n_vertices = -1;  // total number of vertices
    long n_dof = -1;  // number of degrees of freedom
    long seed = -1;  // random number generator seed
    long dim_grid = -1, dim_block = -1, dim_vertex_grid = -1;  // dimensions for the CUDA kernels

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
    thrust::host_vector<T> getArray(const std::string& array_name);

    /**
     * @brief Set a device array from a host array.
     * 
     * @tparam T The type of the array.
     * @param array_name The name of the array.
     * @param host_array The host array.
     */
    template <typename T>
    void setArray(const std::string& array_name, const thrust::host_vector<T>& host_array);

    // ----------------------------------------------------------------------
    // -------------------- Universally Defined Methods ---------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Get the type name of the particle.
     * 
     * @return The type name of the particle.   
     */
    std::string getTypeName() const;

    /**
     * @brief Set the seed for the random number generator.
     * 
     * @param seed The seed for the random number generator.
     */
    void setSeed(long seed);

    /**
     * @brief Set the dimensions for the CUDA kernels and synchronize to the device constant memory.
     * 
     * @param dim_block The number of threads in the block (default is 256).
     */
    void setKernelDimensions(long dim_block = 256);

    /**
     * @brief Synchronize the kernel dimensions to the device constant memory.
     */
    void syncKernelDimensions();

    /**
     * @brief Set the number of particles and synchronize to the device constant memory.
     * 
     * @param n_particles The number of particles.
     */
    void setNumParticles(long n_particles);

    /**
     * @brief Set the degrees of freedom.  Specific values depend on the derived class.
     */
    void setDegreesOfFreedom();

    /**
     * @brief Synchronize the number of particles to the device constant memory.
     */
    void syncNumParticles();

    /**
     * @brief Set the number of vertices for the particles and synchronize to the device constant memory.
     * 
     * @param n_vertices The number of vertices.
     */
    void setNumVertices(long n_vertices);

    /**
     * @brief Synchronize the number of vertices to the device constant memory.
     */
    void syncNumVertices();

    /**
     * @brief Set the number of particles and vertices and synchronize to the device constant memory.
     * Initialize the arrays to the correct sizes depending on the number of particles and vertices.
     * Depends on the derived class to define the logic.
     * 
     * @param n_particles The number of particles.
     * @param n_vertices The number of vertices.
     */
    void setParticleCounts(long n_particles, long n_vertices);

    /**
     * @brief Initialize the dynamic variables and their pointers given the number of particles and dimensions.
     * Depends on the derived class to define the logic.
     */
    void initDynamicVariables();

    /**
     * @brief Clear the dynamic variables and their pointers from the device memory.
     * Depends on the derived class to define the logic.
     */
    void clearDynamicVariables();

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
    void syncNeighborList();
    
    /**
     * @brief Set the energy scales for the particles.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param e The energy scale.
     * @param which The type of energy scale to set ("c", "a", "b", or "l").
     */
    void setEnergyScale(double e, std::string which);

    double getEnergyScale(std::string which);

    /**
     * @brief Set the energy scales for the particles.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param e_c The contact energy scale.
     * @param e_a The area energy scale.
     * @param e_b The bending energy scale.
     * @param e_l The length energy scale.
     */
    void setAllEnergyScales(double e_c, double e_a, double e_b, double e_l);

    /**
     * @brief Set the exponents for the energy terms.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param n The exponent.
     * @param which The type of exponent to set ("c", "a", "b", or "l").
     */
    void setExponent(double n, std::string which);

    double getExponent(std::string which);

    /**
     * @brief Set the exponents for the energy terms.
     * V = e / n (1 - r / sigma) ^ n
     * 
     * @param n_c The contact exponent.
     * @param n_a The area exponent.
     * @param n_b The bending exponent.
     * @param n_l The length exponent.
     */
    void setAllExponents(double n_c, double n_a, double n_b, double n_l);

    /**
     * @brief Initialize the box variables to a desired packing fraction.
     * 
     * @param packing_fraction The desired packing fraction of the box.
     */
    void initializeBox(double packing_fraction);

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
    void setRandomPositions();

    /**
     * @brief Set the particle velocities to random normal values with a given temperature.
     * 
     * @param temperature The desired temperature of the system.
     */
    void setRandomVelocities(double temperature);

    /**
     * @brief Get the diameter of the particles.
     * 
     * @param which The type of diameter to get ("min" for minimum, "max" for maximum, or "avg" for average).
     * @return The diameter of the particles.
     */
    double getDiameter(std::string which = "min");

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
    void scaleToPackingFraction(double packing_fraction);

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
     * Depends on the derived class to define the logic.
     */
    void initGeometricVariables() {};

    /**
     * @brief Clear the geometric variables and their pointers for the object - default is empty.
     * Depends on the derived class to define the logic.
     */
    void clearGeometricVariables() {};

    /**
     * @brief Apply an affine transformation to the positions of the particles.
     * 
     * @param scale_factor The scaling factor.
     */
    void scalePositions(double scale_factor);

    /**
     * @brief Update the positions of the particles using an explicit Euler method.
     * x(t+dt) = x(t) + dt * v(t)
     * 
     * @param dt The time step.
     */
    void updatePositions(double dt);

    /**
     * @brief Update the velocities of the particles using an explicit Euler method.
     * 
     * @param dt The time step.
     */
    void updateVelocities(double dt);

    /**
     * @brief Get the maximum displacement of the particles since the last neighbor list update.
     * 
     * @return The maximum displacement.
     */
    double getMaxDisplacement();

    /**
     * @brief Check if the neighbor list of the particles needs to be updated (if maximum displacement is greater than the neighbor cutoff).
     */
    void checkForNeighborUpdate();

    /**
     * @brief Calculate the neighbor list for the particles.
     */
    void updateNeighborList();

    /**
     * @brief Set the neighbor cutoff for the particles based on some multiple of a defined length scale.
     * Length scale depends on the derived class, defaults to the minimum particle diameter.
     * 
     * @param neighbor_cutoff_multiplier The multiplier for the neighbor cutoff.
     */
    void setNeighborCutoff(double neighbor_cutoff_multiplier);

    /**
     * @brief Print the neighbor list for the particles.  Useful for debugging.
     */
    void printNeighborList();

    /**
     * @brief Zero out the force and potential energy arrays.
     */
    void zeroForceAndPotentialEnergy();

    /**
     * @brief Remove the mean velocity of the particles along each dimension.
     */
    void removeMeanVelocities();

    /**
     * @brief Scale the velocities of the particles to a desired temperature.
     * 
     * @param temperature The desired temperature.
     */
    void scaleVelocitiesToTemperature(double temperature);

    /**
     * @brief Calculate the temperature of the particles.
     * T = sum(KE) * 2 / dof
     */
    double calculateTemperature();

    double getTimeUnit();

    void setMass(double mass);

    // ----------------------------------------------------------------------
    // ---------------------- Pure Virtual Methods --------------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Virtual method to get the total area of the particles.
     * 
     * @return The total area of the particles.
     */
    double getArea() const {
        return static_cast<const Derived*>(this)->getArea();
    }

    /**
     * @brief Virtual method to calculate the ratio of the particle overlap area to the total area.
     * 
     * @return The overlap fraction of the particles.
     */
    double getOverlapFraction() const {
        return static_cast<const Derived*>(this)->getOverlapFraction();
    }

    /**
     * @brief Virtual method to calculate the forces on the particles.
     */
    void calculateForces() {
        static_cast<Derived*>(this)->calculateForces();
    }

    /**
     * @brief Virtual method to calculate the kinetic energy of the particles.
     */
    void calculateKineticEnergy() {
        static_cast<Derived*>(this)->calculateKineticEnergy();
    }
};

#endif /* PARTICLE_H */
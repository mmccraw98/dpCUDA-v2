#ifndef PARTICLE_H
#define PARTICLE_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "config.h"

#include "../../include/data/data_1d.h"
#include "../../include/data/data_2d.h"
#include "../../include/data/array_data.h"

#include <unordered_map>
#include <any>
#include <set>
#include <typeinfo>
#include <iostream>
#include <string>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

struct KernelConfig {
    long dim_grid;
    long dim_block;
};

/**
 * @brief Base class for all particle types.
 */
class Particle {
public:
    Particle();
    virtual ~Particle();  // Ensure virtual destructor for proper cleanup in derived classes

    virtual void initializeFromConfig(const BaseParticleConfig& config);

    std::unique_ptr<BaseParticleConfig> config;

    // Function pointer for the neighbor list update method
    void (Particle::*initNeighborListPtr)();
    void (Particle::*updateNeighborListPtr)();
    void (Particle::*checkForNeighborUpdatePtr)();

    // These arrays (and the parameters) have to be saved to be able to restart from a configuration - all other values can be derived if not defined
    std::vector<std::string> fundamental_values = {"d_positions", "d_velocities"};
    // These are the values that need to be calculated before the log value is calculated
    std::vector<std::string> pre_req_calculations = {"KE", "T", "kinetic_energy"};

    // this is going to be done separately for each derived class
    // the key is the name of the log variable and the value is the function that needs to be called to calculate the log variable
    // i.e. if we want total energy, we need to calculate kinetic energy first
    // many other log variables may depend on the same calculation so we keep track of which ones have been calculated
    std::unordered_map<std::string, std::vector<std::string>> calculation_dependencies = {  // replicate this for each derived class
        {"TE", {"calculate_kinetic_energy"}},
        {"T", {"calculate_kinetic_energy"}},
        {"KE", {"calculate_kinetic_energy"}},  // total kinetic energy scalar
        {"kinetic_energy", {"calculate_kinetic_energy"}}  // kinetic energy array
        // can have nested dependencies i.e. {"particle_KE", {"calculate_particle_kinetic_energy"}}, {"calculate_particle_kinetic_energy", {"calculate_particle_velocities"}}
    };
    virtual void handle_calculation_for_single_dependency(std::string dependency_calculation_name);  // replicate this for each derived class
    std::vector<std::string> reorder_arrays = {"static_particle_index"};  // possibly need to replicate for each derived class - tracks the arrays used to index particle level data
    std::set<std::string> unique_dependents;
    std::set<std::string> unique_dependencies;
    std::map<std::string, bool> dependency_status;
    std::set<std::string> get_unique_dependencies() { return unique_dependencies; }
    std::set<std::string> get_unique_dependents() { return unique_dependents; }
    std::map<std::string, bool> get_dependency_status() { return dependency_status; }
    void define_unique_dependencies();
    void reset_dependency_status();
    void calculate_dependencies(std::string log_name);



    // Device vectors for particle data
    Data1D<double> box_size;
    SwapData2D<double> positions;
    SwapData2D<double> velocities;
    SwapData2D<double> forces;
    Data2D<double> last_neigh_positions;
    Data2D<double> last_cell_positions;
    Data1D<double> neigh_displacements_sq;
    Data1D<double> cell_displacements_sq;
    SwapData1D<double> radii;
    SwapData1D<double> masses;
    Data1D<double> kinetic_energy;
    Data1D<double> potential_energy;
    Data1D<long> neighbor_list;
    Data1D<long> num_neighbors;
    Data1D<long> cell_index;
    Data1D<long> particle_index;
    Data1D<long> static_particle_index;
    Data1D<long> cell_start;

    long num_rebuilds = 0;
    bool switched = false;
    bool using_cell_list = false;
    std::string neighbor_list_update_method = "none";

    // Simulation parameters
    double e_c = -1, e_a = -1, e_b = -1, e_l = -1;  // energy scales for interaction, area, bending, and length
    double n_c = -1, n_a = -1, n_b = -1, n_l = -1;  // exponents for the energy terms
    double neighbor_cutoff = -1;  // cutoff distance for the neighbor list
    double neighbor_displacement_threshold_sq = -1;  // displacement threshold squared after which the neighbor list is updated
    double cell_displacement_threshold_sq = -1;  // displacement threshold squared after which the cell list is updated
    long max_neighbors = -1;  // maximum number of neighbors
    long max_neighbors_allocated = -1;  // maximum number of neighbors allocated for each particle
    long n_particles = -1;  // total number of particles
    long n_vertices = -1;  // total number of vertices
    long n_dof = -1;  // number of degrees of freedom
    long seed = -1;  // random number generator seed

    long particle_dim_grid = -1, particle_dim_block = -1, vertex_dim_grid = -1, vertex_dim_block = -1, cell_dim_grid = -1, cell_dim_block = -1;  // dimensions for the CUDA kernels
    
    long n_cells = -1;  // number of cells in the simulation box
    long n_cells_dim = -1;  // number of cells in each dimension
    double cell_size = -1;  // size of the cells


    ArrayData getArrayData(const std::string& array_name);

    // ----------------------------------------------------------------------
    // -------------------- Universally Defined Methods ---------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Set the neighbor list update method for the particles.
     * 
     * @param method_name The name of the method to use for updating the neighbor list: "cell", "verlet", or "none".
     */
    void setNeighborMethod(std::string method_name);

    /**
     * @brief Set the seed for the random number generator.
     * 
     * @param seed The seed for the random number generator.
     */
    void setSeed(long seed);

    /**
     * @brief Set the dimensions for the CUDA kernels and synchronize to the device constant memory.
     * 
     * @param particle_dim_block The number of threads in the block (default is 256).
     */
    virtual void setKernelDimensions(long particle_dim_block = 256);

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
    virtual void setDegreesOfFreedom();

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
    virtual void setParticleCounts(long n_particles, long n_vertices);

    /**
     * @brief Initialize the dynamic variables and their pointers given the number of particles and dimensions.
     * Depends on the derived class to define the logic.
     */
    virtual void initDynamicVariables();

    /**
     * @brief Clear the dynamic variables and their pointers from the device memory.
     * Depends on the derived class to define the logic.
     */
    virtual void clearDynamicVariables();

    /**
     * @brief Set the box size vector constant in the device memory.
     * 
     * @param box_size The box size vector.
     */
    void setBoxSize(const thrust::host_vector<double>& host_box_size);

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
     * @brief Get the energy scale for the particles.
     * 
     * @param which The type of energy scale to get ("c", "a", "b", or "l").
     * @return The energy scale.
     */
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

    /**
     * @brief Get the exponent for the energy terms.
     * 
     * @param which The type of exponent to get ("c", "a", "b", or "l").
     * @return The exponent.
     */
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
     * @param seed_offset The offset for the random number generator seed.
     */
    void setRandomUniform(thrust::device_vector<double>& values, double min, double max, long seed_offset = 0);

    /**
     * @brief Fill a device vector with random normal values.
     * 
     * @param values The device vector.
     * @param mean The mean value.
     * @param stddev The standard deviation.
     */
    void setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev, long seed_offset = 0);

    /**
     * @brief Set the particle positions to random uniform values within the box size.
     */
    virtual void setRandomPositions();

    /**
     * @brief Set the particle velocities to random normal values with a given temperature.
     * 
     * @param temperature The desired temperature of the system.
     */
    virtual void setRandomVelocities(double temperature);

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
     * Depends on the derived class to define the logic.
     */
    virtual void initGeometricVariables() {};

    /**
     * @brief Clear the geometric variables and their pointers for the object - default is empty.
     * Depends on the derived class to define the logic.
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
     * @brief Get the maximum squared displacement of the particles since the last neighbor list update.
     * 
     * @return The maximum squared displacement.
     */
    virtual double getMaxSquaredNeighborDisplacement();

    /**
     * @brief Get the maximum squared displacement of the particles since the last cell list update.
     * 
     * @return The maximum squared displacement.
     */
    virtual double getMaxSquaredCellDisplacement();

    /**
     * @brief Check if the neighbor list of the particles needs to be updated (if maximum squared displacement is greater than the neighbor cutoff squared).
     */
    virtual void checkForNeighborUpdate();

    /**
     * @brief Check if the neighbor list of the particles needs to be updated using the Verlet criterion.
     */
    virtual void checkForVerletListUpdate();

    /**
     * @brief Check if the cell list of the particles needs to be updated using the Verlet criterion.
     */
    virtual void checkForCellListUpdate();

    /**
     * @brief Calculate the neighbor list for the particles.
     */
    virtual void updateVerletList();

    /**
     * @brief Initialize and calculate the neighbor list for the particles.
     */
    virtual void initNeighborList();

    virtual void clearNeighborVariables();

    virtual void initAllToAllListVariables();

    virtual void initAllToAllList();

    virtual void checkForAllToAllUpdate();

    virtual void initVerletListVariables();

    virtual void initCellListVariables();

    virtual void initVerletList();

    /**
     * @brief Initialize and calculate the cell list for the particles.
     */
    virtual void initCellList();

    /**
     * @brief Synchronize the cell list sizes to the device constant memory.
     */
    virtual void syncCellList();

    /**
     * @brief Update the cell list for the particles.
     */
    virtual void updateCellList();

    /**
     * @brief Reorder the particle data in global memory to match the cell list.
     */
    virtual void reorderParticleData();

    /**
     * @brief Update the neighbor list using the cell list.
     */
    virtual void updateCellNeighborList();

    /**
     * @brief Set the neighbor cutoff for the particles based on some multiple of a defined length scale.
     * Length scale depends on the derived class, defaults to the minimum particle diameter.
     * 
     * @param neighbor_cutoff_multiplier The multiplier for the neighbor cutoff.
     * @param neighbor_displacement_multiplier The multiplier for the neighbor displacement.
     */
    virtual bool setNeighborSize(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier);

    /**
     * @brief Set the cell size for the cell list.
     * 
     * @param cell_size_multiplier The multiplier for the cell size.
     * @param cell_displacement_multiplier The multiplier for the cell displacement.
     */
    virtual bool setCellSize(double num_particles_per_cell, double cell_displacement_multiplier);

    virtual double getNumberDensity();

    /**
     * @brief Zero out the force and potential energy arrays.
     */
    virtual void zeroForceAndPotentialEnergy();

    /**
     * @brief Remove the mean velocity of the particles along each dimension.
     */
    virtual void removeMeanVelocities();

    /**
     * @brief Scale the velocities of the particles to a desired temperature.
     * 
     * @param temperature The desired temperature.
     */
    virtual void scaleVelocitiesToTemperature(double temperature);

    /**
     * @brief Calculate the temperature of the particles.
     * T = sum(KE) * 2 / dof
     */
    virtual double calculateTemperature();

    /**
     * @brief Get the time unit of the simulation.  Default is sigma * sqrt(mass / epsilon)
     * 
     * @return The time unit of the simulation.
     */
    virtual double getTimeUnit();

    /**
     * @brief Uniformly set the mass of the particles.
     * 
     * @param mass The mass of the particles.
     */
    virtual void setMass(double mass);

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
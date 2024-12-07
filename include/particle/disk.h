#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "particle.h"
#include "config.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

/**
 * @brief Configuration for a bidisperse system of disk particles.
 */
struct BidisperseDiskConfig : public BidisperseParticleConfig {

    /**
     * @brief Constructor for the bidisperse disk configuration.
     * 
     * @param seed The seed for the random number generator.
     * @param n_particles The number of particles.
     * @param mass The mass of the particles.
     * @param e_c The energy scale for the interaction potential.
     * @param n_c The exponent for the interaction potential.
     * @param packing_fraction The packing fraction of the particles.
     * @param neighbor_cutoff_multiplier Particles within this multiple of the maximum particle diameter will be considered neighbors.
     * @param neighbor_displacement_multiplier If the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated.
     * @param num_particles_per_cell The desired number of particles per cell.
     * @param cell_displacement_multiplier The multiplier for the cell displacement in terms of the cell size.
     * @param neighbor_list_update_method The method for updating the neighbor list.
     * @param particle_dim_block The number of threads in the block.
     * @param size_ratio The ratio of the sizes of the two particle types.
     * @param count_ratio The ratio of the counts of the two particle types.
     */
    BidisperseDiskConfig(long seed, long n_particles, double mass, double e_c, double n_c,
                            double packing_fraction, double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier, double num_particles_per_cell, double cell_displacement_multiplier, std::string neighbor_list_update_method, long particle_dim_block,
                            double size_ratio, double count_ratio)
        : BidisperseParticleConfig(seed, n_particles, mass, e_c, n_c, packing_fraction, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, neighbor_list_update_method, particle_dim_block, size_ratio, count_ratio) {
            type_name = "Disk";
        }
};


/**
 * @brief Soft repulsive disk particle class.
 */
class Disk : public Particle {
public:
    Disk();

    virtual ~Disk();  // TODO: may want to remove virtual

    // ----------------------------------------------------------------------
    // --------------------- Overridden Methods -----------------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Set the dimensions for the CUDA kernels.
     * 
     * @param particle_dim_block The number of threads in the block (default is 256).
     */
    void setKernelDimensions(long particle_dim_block = 256) override;

    // ----------------------------------------------------------------------
    // ------------- Implementation of Pure Virtual Methods -----------------
    // ----------------------------------------------------------------------

    /**
     * @brief Get the total area of the particles by summing the squares of the radii.
     * 
     * @return The total area of the particles.
     */
    double getParticleArea() const override;

    /**
     * @brief Get the fraction of the area involving the overlap between particles using the lense formula.
     * 
     * @return The overlap fraction of the particles.
     */
    double getOverlapFraction() const override;
    
    /**
     * @brief Calculate the forces and potential energies of the particles.
     * V = e / n * (1 - r / sigma) ^ n
     * 
     */
    void calculateForces() override;

    /**
     * @brief Calculate the kinetic energy of the particles.
     */
    void calculateKineticEnergy();

    void calculateForceDistancePairs() override;

    void calculateWallForces() override;
};

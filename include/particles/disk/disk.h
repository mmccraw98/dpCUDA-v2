#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particles/base/particle.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nlohmann/json.hpp>

#include "../../include/particles/disk/config.h"


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

    void loadData(const std::string& root) override;
};

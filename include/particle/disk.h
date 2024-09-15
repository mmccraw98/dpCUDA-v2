#ifndef DISK_H
#define DISK_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "particle.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Disk : public Particle {
public:
    Disk();

    virtual ~Disk();

    // ----------------------------------------------------------------------
    // --------------------- Overridden Methods -----------------------------
    // ----------------------------------------------------------------------

    /**
     * @brief Set the dimensions for the CUDA kernels.
     * 
     * @param dim_block The number of threads in the block (default is 256).
     */
    void setKernelDimensions(long dim_block = 256) override;

    // ----------------------------------------------------------------------
    // ------------- Implementation of Pure Virtual Methods -----------------
    // ----------------------------------------------------------------------
    
    /**
     * @brief Get the total area of the particles by summing the squares of the radii.
     * 
     * @return The total area of the particles.
     */
    double getArea() const override;

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
    void calculateKineticEnergy() override;
};

#endif /* DISK_H */

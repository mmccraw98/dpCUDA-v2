#ifndef DISK_H
#define DISK_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/kernels/kernels.cuh"
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
     * @brief Calculate the forces on the particles.
     */
    void calculateForces() override;

    /**
     * @brief Calculate the kinetic energy of the particles.
     */
    void calculateKineticEnergy() override;
};

#endif /* DISK_H */

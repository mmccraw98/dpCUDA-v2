#ifndef DISK_H
#define DISK_H

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


struct BidisperseDiskConfig : public BidisperseParticleConfig {
    std::string type_name = "Disk";

    BidisperseDiskConfig(long seed, long n_particles, double mass, double e_c, double n_c,
                            double packing_fraction, double neighbor_cutoff, long dim_block,
                            double size_ratio, double count_ratio)
        : BidisperseParticleConfig(seed, n_particles, mass, e_c, n_c, packing_fraction, neighbor_cutoff, dim_block, size_ratio, count_ratio) {}

    nlohmann::json to_json() const override {
        nlohmann::json j = BidisperseParticleConfig::to_json();  // Call base class serialization
        j["type_name"] = type_name;
        return j;
    }
};

class Disk : public Particle {
public:
    Disk();

    virtual ~Disk();

    std::string type_name = "Disk";

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
    void calculateKineticEnergy();
};

#endif /* DISK_H */

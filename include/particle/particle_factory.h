#ifndef PARTICLE_FACTORY_H
#define PARTICLE_FACTORY_H

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "particle.h"
#include "config.h"
#include "disk.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename ConfigType>
std::unique_ptr<Particle> create_particle(const ConfigType& config) {
    static_assert(std::is_base_of<BaseParticleConfig, ConfigType>::value, "ConfigType must derive from BaseParticleConfig");

    if (config.type_name == "Disk") {
        std::cout << "create_particle: Creating Disk particle..." << std::endl;
        auto particle = std::make_unique<Disk>();
        // particle->config = config;
        particle->setSeed(config.seed);
        // set/sync number of vertices/particles, define the array sizes
        particle->setParticleCounts(config.n_particles, 0);
        // set/sync kernel dimensions
        particle->setKernelDimensions(config.dim_block);

        // Handle the dispersity type
        if constexpr (std::is_same<ConfigType, BidisperseDiskConfig>::value) {
            if (config.dispersity_type == "bidisperse") {
                // define the particle sizes, initialize the box to a set packing fraction, and set random positions
                particle->setBiDispersity(config.size_ratio, config.count_ratio);
                particle->initializeBox(config.packing_fraction);
            }
        }

        particle->setRandomPositions();
        // Define geometry when relevant (i.e. initialize vertex configurations, calculate shape parameters, etc.)

        // set/sync energies
        particle->setEnergyScale(config.e_c, "c");
        particle->setExponent(config.n_c, "c");
        particle->setMass(config.mass);
        // define the neighbor cutoff size
        particle->setNeighborCutoff(config.neighbor_cutoff);  // 1.5 * min_diameter
        // update the neighbor list
        particle->updateNeighborList();
        return particle;
    }
    throw std::invalid_argument("Invalid particle type: " + config.type_name);
}

#endif /* PARTICLE_FACTORY_H */

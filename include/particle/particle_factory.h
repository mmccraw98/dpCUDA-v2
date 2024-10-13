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
#include <type_traits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Generic factory function to create particles from any config type
template <typename ConfigType>
std::unique_ptr<Particle<typename ConfigType::ParticleType>> create_particle(const ConfigType& config) {
    static_assert(std::is_base_of<BaseParticleConfig, ConfigType>::value, 
                  "ConfigType must derive from BaseParticleConfig");

    if (config.type_name == "Disk") {
        auto particle = std::make_unique<Disk>();  // Create Disk
        particle->initialize(config);              // Initialize with config
        return particle;                           // Return as std::unique_ptr<Particle>
    }
    
    // Add other particle types as needed
    throw std::invalid_argument("Unsupported particle type: " + config.type_name);
}


#endif /* PARTICLE_FACTORY_H */

#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"
#include "base/particle.h"
#include "config.h"
#include "disk/disk.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @brief Create a particle from an arbitrary configuration struct.
 * 
 * @tparam ConfigType The type of the configuration.
 * @param config The configuration.
 * @return std::unique_ptr<Particle> The particle.
 */
template <typename ConfigType>
std::unique_ptr<Particle> create_particle(const ConfigType& config) {
    static_assert(std::is_base_of<BaseParticleConfig, ConfigType>::value, "ConfigType must derive from BaseParticleConfig");

    if (config.type_name == "Disk") {
        std::cout << "create_particle: Creating Disk particle..." << std::endl;
        auto particle = std::make_unique<Disk>();
        particle->initializeFromConfig(config);
        return particle;
    }
    throw std::invalid_argument("Invalid particle type: " + config.type_name);
}

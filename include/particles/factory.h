#pragma once

#include "particles/base/particle.h"
#include "particles/disk/disk.h"
#include "particles/rigid_bumpy/rigid_bumpy.h"
#include "particles/standard_configs.h"

#include "io/io_utils.h"

inline std::unique_ptr<Particle> createParticle(ConfigDict& config, bool minimize = false) {
    std::string particle_type = config.at("particle_type").get<std::string>();
    std::unique_ptr<Particle> particle;
    if (particle_type == "Disk") {
        particle = std::make_unique<Disk>();
    } else if (particle_type == "RigidBumpy") {
        particle = std::make_unique<RigidBumpy>();
    }
    particle->initializeFromConfig(config, minimize);
    return particle;
}

// overload to define the number of particles, the packing fraction, and the name of the particle type
inline std::unique_ptr<Particle> createParticle(long n_particles, double packing_fraction, std::string particle_type, bool minimize = false) {
    ConfigDict config;
    if (particle_type == "Disk") {
        config = get_standard_disk_config(n_particles, packing_fraction);
    } else if (particle_type == "RigidBumpy") {
        config = get_standard_rigid_bumpy_config(n_particles, packing_fraction, true);
    } else if (particle_type == "RigidBumpyNoRotation") {
        config = get_standard_rigid_bumpy_config(n_particles, packing_fraction, false);
    }
    return createParticle(config, minimize);
}
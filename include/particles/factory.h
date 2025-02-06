#pragma once

#include "particles/base/particle.h"
#include "particles/disk/disk.h"
#include "particles/rigid_bumpy/rigid_bumpy.h"

#include "io/io_utils.h"

inline std::unique_ptr<Particle> createParticle(ConfigDict& config, bool minimize = false) {
    std::string type_name = config.at("type_name").get<std::string>();
    std::unique_ptr<Particle> particle;
    if (type_name == "Disk") {
        particle = std::make_unique<Disk>();
    } else if (type_name == "RigidBumpy") {
        particle = std::make_unique<RigidBumpy>();
    }
    particle->initializeFromConfig(config, minimize);
    return particle;
}
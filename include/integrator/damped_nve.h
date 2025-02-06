#pragma once

#include "integrator.h"
#include "../particles/base/particle.h"

inline ConfigDict get_damped_nve_config_dict(double dt, double damping_coefficient) {
    ConfigDict config;
    config["integrator_type"] = "DampedNVE";
    config["dt"] = dt;
    config["damping_coefficient"] = damping_coefficient;
    return config;
}

class DampedNVE : public Integrator {
public:

    DampedNVE(Particle& particle, ConfigDict& config);
    ~DampedNVE();

    double dt;  // time step
    double damping_coefficient;
    
    void step() override;

    void wall_step();
};

#pragma once

#include "integrator.h"
#include "../particles/base/particle.h"

inline ConfigDict get_nve_config_dict(double dt) {
    ConfigDict config;
    config["integrator_type"] = "NVE";
    config["dt"] = dt;
    return config;
}

/**
 * @brief NVE integrator class.
 * 
 * This class implements a Velocity-Verlet NVE integrator for the particle system.
 */
class NVE : public Integrator {
public:
    /**
     * @brief Constructor for NVE class.
     * 
     * @param particle Reference to Particle object.
     * @param config Reference to NVEConfig object.
     */
    NVE(Particle& particle, ConfigDict& config);
    ~NVE();

    double dt;  // time step
    
    /**
     * @brief Advance the state of the system by one time step using the Velocity-Verlet algorithm.
     */
    void step() override;

    void wall_step();
};

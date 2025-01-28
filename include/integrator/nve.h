#pragma once

#include "integrator.h"
#include "../particles/base/particle.h"

/**
 * @brief Configuration for the NVE integrator.
 * 
 * This class extends the IntegratorConfig class and adds a dt member variable.
 */
class NVEConfigDict : public IntegratorConfigDict {
public:
    NVEConfigDict() {
        insert("integrator_type", "NVE");
        insert("dt", 0.0);
    }
};


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
    NVE(Particle& particle, const NVEConfigDict& config);
    ~NVE();

    double dt;  // time step
    
    /**
     * @brief Advance the state of the system by one time step using the Velocity-Verlet algorithm.
     */
    void step() override;

    void wall_step();
};

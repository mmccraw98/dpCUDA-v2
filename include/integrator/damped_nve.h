#pragma once

#include "integrator.h"
#include "../particles/base/particle.h"

struct DampedNVEConfigDict : public IntegratorConfigDict {
public:
    DampedNVEConfigDict() {
        insert("integrator_type", "DampedNVE");
        insert("dt", 0.0);
        insert("damping_coefficient", 0.0);
    }
};


class DampedNVE : public Integrator {
public:

    DampedNVE(Particle& particle, const DampedNVEConfigDict& config);
    ~DampedNVE();

    double dt;  // time step
    double damping_coefficient;
    
    void step() override;

    void wall_step();
};

#ifndef NVE_H
#define NVE_H

#include "integrator.h"
#include "../particle/particle.h"

// nve has to be templated since it can accept any particle type

template <typename Derived>
class NVE : public Integrator<Derived> {
public:
    NVE(Particle<Derived>& particle, double dt) : Integrator<Derived>(particle), dt(dt) {}
    ~NVE() {}

    double dt;
    
    void step() {
        this->particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t) + dt / 2 * f(t) / m
        this->particle.updatePositions(dt);  // x(t+dt) = x(t) + dt * v(t+dt)
        
        // TODO: may want to consider moving the zero-out and neigh-update into the force calc
        this->particle.zeroForceAndPotentialEnergy();
        this->particle.checkForNeighborUpdate();
        this->particle.calculateForces();
        
        this->particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t+dt) + dt / 2 * f(t+dt) / m
    }
};

#endif /* NVE_H */

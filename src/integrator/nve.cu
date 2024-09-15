#include "../../include/integrator/nve.h"
#include "../../include/particle/particle.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// Constructor for NVEIntegrator
NVE::NVE(Particle& particle, double dt) : Integrator(particle), dt(dt) {
}

NVE::~NVE() {

}

// NVE step function implementation
void NVE::step() {
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t) + dt / 2 * f(t) / m
    particle.updatePositions(dt);  // x(t+dt) = x(t) + dt * v(t+dt)
    
    // TODO: may want to consider moving the zero-out and neigh-update into the force calc
    particle.zeroForceAndPotentialEnergy();
    particle.checkForNeighborUpdate();
    particle.calculateForces();
    
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t+dt) + dt / 2 * f(t+dt) / m
}

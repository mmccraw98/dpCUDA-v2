#include "../../include/integrator/damped_nve.h"
#include "../../include/particles/base/particle.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>

DampedNVE::DampedNVE(Particle& particle, const DampedNVEConfig& config) : Integrator(particle, config), dt(config.dt), damping_coefficient(config.damping_coefficient) {
}

DampedNVE::~DampedNVE() {

}

// make get/set methods to sync values to device when relevant

void DampedNVE::step() {
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t) + dt / 2 * f(t) / m
    particle.updatePositions(dt);  // x(t+dt) = x(t) + dt * v(t+dt)

    // TODO: may want to consider moving the neigh-update into the force calc since it is essentially part of the force calculation
    particle.checkForNeighborUpdate();
    particle.zeroForceAndPotentialEnergy();
    particle.calculateDampedForces(damping_coefficient);
    particle.calculateForces();
    
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t+dt) + dt / 2 * f(t+dt) / m
}

void DampedNVE::wall_step() {
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t) + dt / 2 * f(t) / m
    particle.updatePositions(dt);  // x(t+dt) = x(t) + dt * v(t+dt)

    // TODO: may want to consider moving the neigh-update into the force calc since it is essentially part of the force calculation
    particle.checkForNeighborUpdate();
    particle.zeroForceAndPotentialEnergy();
    particle.calculateDampedForces(damping_coefficient);
    particle.calculateForces();
    
    particle.updateVelocities(0.5 * dt);  // v(t+dt) = v(t+dt) + dt / 2 * f(t+dt) / m
}
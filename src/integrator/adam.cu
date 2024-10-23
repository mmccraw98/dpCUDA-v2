#include "../../include/integrator/adam.h"

Adam::Adam(Particle& particle, const AdamConfig& config) : Integrator(particle, config), alpha(config.alpha), beta1(config.beta1), beta2(config.beta2), epsilon(config.epsilon) {

}

Adam::~Adam() {

}

void Adam::step(long step) {
    // TODO: may want to consider moving the zero-out and neigh-update into the force calc
    particle.zeroForceAndPotentialEnergy();  // not needed since forces are calculated in place
    particle.checkForNeighborUpdate();
    particle.calculateForces();

    particle.updatePositionsAdam(step, alpha, beta1, beta2, epsilon);
}
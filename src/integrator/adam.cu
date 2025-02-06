#include "../../include/integrator/adam.h"

Adam::Adam(Particle& particle, ConfigDict& config) : Integrator(particle, config), alpha(config.at("alpha").get<double>()), beta1(config.at("beta1").get<double>()), beta2(config.at("beta2").get<double>()), epsilon(config.at("epsilon").get<double>()) {

}

Adam::~Adam() {

}

void Adam::minimize(long step) {
    // TODO: may want to consider moving the zero-out and neigh-update into the force calc
    particle.zeroForceAndPotentialEnergy();  // not needed since forces are calculated in place
    particle.checkForNeighborUpdate();
    particle.calculateForces();

    particle.updatePositionsAdam(step, alpha, beta1, beta2, epsilon);
}
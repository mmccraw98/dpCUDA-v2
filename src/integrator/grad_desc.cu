#include "../../include/integrator/grad_desc.h"

GradDesc::GradDesc(Particle& particle, const GradDescConfigDict& config) : Integrator(particle, config), alpha(config["alpha"]) {

}

GradDesc::~GradDesc() {

}

void GradDesc::step() {
    particle.zeroForceAndPotentialEnergy();
    particle.checkForNeighborUpdate();
    particle.calculateForces();

    particle.updatePositionsGradDesc(alpha);
}
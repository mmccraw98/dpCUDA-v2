#include "../../include/integrator/grad_desc.h"

GradDesc::GradDesc(Particle& particle, ConfigDict& config) : Integrator(particle, config), alpha(config.at("alpha").get<double>()) {

}

GradDesc::~GradDesc() {

}

void GradDesc::step() {
    particle.zeroForceAndPotentialEnergy();
    particle.checkForNeighborUpdate();
    particle.calculateForces();

    particle.updatePositionsGradDesc(alpha);
}
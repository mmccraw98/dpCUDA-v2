#include "../../include/integrator/fire.h"

Fire::Fire(Particle& particle, ConfigDict& config) : Integrator(particle, config), alpha(config.at("alpha_init").get<double>()), dt(config.at("dt").get<double>()) {
    dt_max = 10.0 * dt;
    dt_min = 0.001 * dt;
    alpha_init = alpha;
}

Fire::~Fire() {

}

void Fire::step() {
    double power = particle.getPowerFire();
    if (power > 0) {  // if moving downhill, increase the inertia
        N_good++;
        N_bad = 0;
        if (N_good > N_min) {
            dt = std::min(dt * f_inc, dt_max);
            alpha *= f_alpha;
        }
    } else {  // if moving uphill
        N_good = 0;
        N_bad++;
        if (N_bad > N_bad_max) {
            stopped = true;
            return;
        }
        dt = std::max(dt * f_dec, dt_min);
        alpha = alpha_init;  // reset the alpha
        particle.updatePositions(- dt / 2.0);  // move the positions back
        particle.setVelocitiesToZero();  // stop the motion
    }

    // velocity verlet with velocity mixing
    particle.updateVelocities(dt / 2.0);  // update the velocities
    particle.mixVelocitiesAndForces(alpha);  // mix the velocities and forces
    particle.updatePositions(dt);

    // TODO: may want to consider moving the zero-out and neigh-update into the force calc
    particle.zeroForceAndPotentialEnergy();  // not needed since forces are calculated in place
    particle.checkForNeighborUpdate();
    particle.calculateForces();
    
    particle.updateVelocities(dt / 2.0);
}
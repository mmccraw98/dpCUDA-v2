#include "../../include/integrator/integrator.h"
#include "../../include/particle/particle.h"

Integrator::Integrator(Particle& particle, const IntegratorConfig& config) : particle(particle), config(config) {}

Integrator::~Integrator() {}

#include "../../include/integrator/integrator.h"
#include "../../include/particles/base/particle.h"

Integrator::Integrator(Particle& particle, ConfigDict& config) : particle(particle), config(config) {}

Integrator::~Integrator() {}

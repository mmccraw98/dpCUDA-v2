#include "../../include/integrator/integrator.h"
#include "../../include/particle/particle.h"

// Constructor and destructor can remain in the .cu file
Integrator::Integrator(Particle& particle) : particle(particle) {
    
}

Integrator::~Integrator() {}

#include "../../include/integrator/nve.h"
#include "../../include/particle/particle.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// Constructor for NVEIntegrator
NVE::NVE(Particle& particle) : Integrator(particle) {
    // Any additional initialization can go here if needed
}

NVE::~NVE() {

}

// NVE step function implementation
void NVE::step() {
    std::cout << "NVE::step" << std::endl;
}

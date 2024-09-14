#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "../particle/particle.h"

class Integrator {
protected:
    Particle& particle;  // Reference to Particle object

public:
    // Constructor accepting a reference to a Particle object
    Integrator(Particle& particle);

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Integrator();

    // Pure virtual function for stepping the integrator
    virtual void step() = 0;
};

#endif /* INTEGRATOR_H */

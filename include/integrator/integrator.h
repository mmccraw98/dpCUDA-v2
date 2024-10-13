#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "../particle/particle.h"

template <typename Derived>
class Integrator {
protected:
    Particle<Derived>& particle;  // Reference to Particle object

public:
    // Constructor accepting a reference to a Particle object
    Integrator(Particle<Derived>& particle) : particle(particle) {};

    // Virtual destructor to ensure proper cleanup in derived classes
    ~Integrator() {}

    // Pure virtual function for stepping the integrator
    void step() {
        static_cast<Derived*>(this)->step();
    }
};

#endif /* INTEGRATOR_H */

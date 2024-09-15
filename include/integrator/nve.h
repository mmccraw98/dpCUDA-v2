#ifndef NVE_H
#define NVE_H

#include "integrator.h"
#include "../particle/particle.h"

class NVE : public Integrator {
public:
    NVE(Particle& particle, double dt);
    ~NVE();

    double dt;
    
    void step() override;
};

#endif /* NVE_H */

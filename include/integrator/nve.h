#ifndef NVE_H
#define NVE_H

#include "integrator.h"
#include "../particle/particle.h"

class NVE : public Integrator {
public:
    NVE(Particle& particle);
    ~NVE();
    
    void step() override;
};

#endif /* NVE_H */

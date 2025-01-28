#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"

struct GradDescConfigDict : public IntegratorConfigDict {
public:
    GradDescConfigDict() {
        insert("integrator_type", "GradDesc");
        insert("alpha", 0.0);
    }
};


class GradDesc : public Integrator {
    public:
        GradDesc(Particle& particle, const GradDescConfigDict& config);
        ~GradDesc();

        double alpha;

        void step();
};
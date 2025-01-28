#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"

struct AdamConfigDict : public IntegratorConfigDict {
public:
    AdamConfigDict() {
        insert("integrator_type", "Adam");
        insert("alpha", 0.0);
        insert("beta1", 0.0);
        insert("beta2", 0.0);
        insert("epsilon", 0.0);
    }
};


class Adam : public Integrator {
    public:
        Adam(Particle& particle, const AdamConfigDict& config);
        ~Adam();

        double alpha;
        double beta1;
        double beta2;
        double epsilon;

        void step() override {}
        void minimize(long step);
};
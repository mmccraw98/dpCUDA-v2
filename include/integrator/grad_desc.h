#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"

inline ConfigDict get_grad_desc_config_dict(double alpha) {
    ConfigDict config;
    config["integrator_type"] = "GradDesc";
    config["alpha"] = alpha;
    return config;
}

class GradDesc : public Integrator {
    public:
        GradDesc(Particle& particle, ConfigDict& config);
        ~GradDesc();

        double alpha;

        void step();
};
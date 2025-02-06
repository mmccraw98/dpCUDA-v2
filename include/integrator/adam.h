#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"


inline ConfigDict get_adam_config_dict(double alpha, double beta1, double beta2, double epsilon) {
    ConfigDict config;
    config["integrator_type"] = "Adam";
    config["alpha"] = alpha;
    config["beta1"] = beta1;
    config["beta2"] = beta2;
    config["epsilon"] = epsilon;
    return config;
}

class Adam : public Integrator {
    public:
        Adam(Particle& particle, ConfigDict& config);
        ~Adam();

        double alpha;
        double beta1;
        double beta2;
        double epsilon;

        void step() override {}
        void minimize(long step);
};
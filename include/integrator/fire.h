#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"


inline ConfigDict get_fire_config_dict(double alpha_init, double dt) {
    ConfigDict config;
    config["integrator_type"] = "Fire";
    config["alpha_init"] = alpha_init;
    config["dt"] = dt;
    return config;
}

class Fire : public Integrator {
    public:
        Fire(Particle& particle, ConfigDict& config);
        ~Fire();

        double alpha;
        double alpha_init;
        double dt;
        double dt_max;
        double dt_min;
        double f_inc = 1.1;
        double f_dec = 0.5;
        double f_alpha = 0.99;
        long N_min = 5;
        long N_good = 0;
        long N_bad = 0;
        long N_bad_max = 10;
        bool stopped = false;

        void step() override;
};
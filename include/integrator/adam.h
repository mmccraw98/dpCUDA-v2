#pragma once

#include "../particle/particle.h"
#include "integrator.h"


class AdamConfig : public IntegratorConfig {
public:
    double alpha;
    double beta1;
    double beta2;
    double epsilon;
    
    AdamConfig(double alpha, double beta1, double beta2, double epsilon) : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon) {
        integrator_type = "Adam";
    }

    void from_json(const nlohmann::json& j) override {
        alpha = j["alpha"];
        beta1 = j["beta1"];
        beta2 = j["beta2"];
        epsilon = j["epsilon"];
    }

    nlohmann::json to_json() const override {
        nlohmann::json j = IntegratorConfig::to_json();
        j["alpha"] = alpha;
        j["beta1"] = beta1;
        j["beta2"] = beta2;
        j["epsilon"] = epsilon;
        return j;
    }
};

class Adam : public Integrator {
    public:
        Adam(Particle& particle, const AdamConfig& config);
        ~Adam();

        double alpha;
        double beta1;
        double beta2;
        double epsilon;

        void step() override {}
        void step(long step);
};
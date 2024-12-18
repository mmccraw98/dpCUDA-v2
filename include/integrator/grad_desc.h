#pragma once

#include "../particles/base/particle.h"
#include "integrator.h"


class GradDescConfig : public IntegratorConfig {
public:
    double alpha;
    
    GradDescConfig(double alpha) : alpha(alpha) {
        integrator_type = "GradDesc";
    }

    void from_json(const nlohmann::json& j) override {
        alpha = j["alpha"];
    }

    nlohmann::json to_json() const override {
        nlohmann::json j = IntegratorConfig::to_json();
        j["alpha"] = alpha;
        return j;
    }
};

class GradDesc : public Integrator {
    public:
        GradDesc(Particle& particle, const GradDescConfig& config);
        ~GradDesc();

        double alpha;

        void step();
};
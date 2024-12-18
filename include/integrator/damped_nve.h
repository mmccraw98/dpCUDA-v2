#pragma once

#include "integrator.h"
#include "../particles/base/particle.h"


class DampedNVEConfig : public IntegratorConfig {
public:
    double dt;  // time step
    double damping_coefficient;

    DampedNVEConfig(double dt, double damping_coefficient) : dt(dt), damping_coefficient(damping_coefficient) {
        integrator_type = "DampedNVE";
    }

    void from_json(const nlohmann::json& j) override {
        dt = j["dt"];
        damping_coefficient = j["damping_coefficient"];
    }

    nlohmann::json to_json() const override {
        nlohmann::json j = IntegratorConfig::to_json();
        j["dt"] = dt;
        j["damping_coefficient"] = damping_coefficient;
        return j;
    }
};


class DampedNVE : public Integrator {
public:

    DampedNVE(Particle& particle, const DampedNVEConfig& config);
    ~DampedNVE();

    double dt;  // time step
    double damping_coefficient;
    
    void step() override;

    void wall_step();
};

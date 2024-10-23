#pragma once

#include "integrator.h"
#include "../particle/particle.h"

/**
 * @brief Configuration for the NVE integrator.
 * 
 * This class extends the IntegratorConfig class and adds a dt member variable.
 */
class NVEConfig : public IntegratorConfig {
public:
    double dt;  // time step

    /**
     * @brief Constructor for NVEConfig class.
     * 
     * @param dt Time step.
     */
    NVEConfig(double dt) : dt(dt) {
        integrator_type = "NVE";
    }

    /**
     * @brief Method to parse the NVEConfig object from a JSON object.
     * 
     * @param j JSON object to parse.
     */
    void from_json(const nlohmann::json& j) override {
        dt = j["dt"];
    }

    /**
     * @brief Method to convert the NVEConfig object to a JSON object.
     * 
     * @return JSON object.
     */
    nlohmann::json to_json() const override {
        nlohmann::json j = IntegratorConfig::to_json();
        j["dt"] = dt;
        return j;
    }
};


/**
 * @brief NVE integrator class.
 * 
 * This class implements a Velocity-Verlet NVE integrator for the particle system.
 */
class NVE : public Integrator {
public:
    /**
     * @brief Constructor for NVE class.
     * 
     * @param particle Reference to Particle object.
     * @param config Reference to NVEConfig object.
     */
    NVE(Particle& particle, const NVEConfig& config);
    ~NVE();

    double dt;  // time step
    
    /**
     * @brief Advance the state of the system by one time step using the Velocity-Verlet algorithm.
     */
    void step() override;
};

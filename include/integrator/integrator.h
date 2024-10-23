#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "../particle/particle.h"

#include <nlohmann/json.hpp>


/**
 * @brief IntegratorConfig class.
 * 
 * This class is used to store the configuration for the integrator object.
 */
struct IntegratorConfig {
    std::string integrator_type;  // the type of integrator to use

    virtual ~IntegratorConfig() = default;

    /**
     * @brief Parse the integrator configuration from a JSON object.
     * 
     * @param j The JSON object to parse.
     */
    virtual void from_json(const nlohmann::json& j) = 0;

    /**
     * @brief Convert the integrator configuration to a JSON object.
     * 
     * @return The JSON object.
     */
    virtual nlohmann::json to_json() const {
        return {{"integrator_type", integrator_type}};
    };
};


/**
 * @brief Integrator class.
 * 
 * Base class for all integrators.
 * An integrator is responsible for updating the state of the system through some means.
 */
class Integrator {
protected:
    Particle& particle;  // Reference to Particle object

public:
    const IntegratorConfig& config;  // Reference to IntegratorConfig object

    /**
     * @brief Constructor for Integrator class.
     * 
     * @param particle Reference to Particle object.
     * @param config Reference to IntegratorConfig object.
     */
    Integrator(Particle& particle, const IntegratorConfig& config);

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Integrator();

    /**
     * @brief Virtual method to perform a single integration step.
     */
    virtual void step() = 0;
};

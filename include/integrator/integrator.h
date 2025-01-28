#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "../particles/base/particle.h"

#include "../include/utils/config_dict.h"

#include <nlohmann/json.hpp>


/**
 * @brief IntegratorConfig class.
 * 
 * This class is used to store the configuration for the integrator object.
 */
struct IntegratorConfigDict : public ConfigDict {
public:
    IntegratorConfigDict() : ConfigDict() {
        insert("integrator_type", "none");
    }
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
    const IntegratorConfigDict& config;  // Reference to IntegratorConfig object

    /**
     * @brief Constructor for Integrator class.
     * 
     * @param particle Reference to Particle object.
     * @param config Reference to IntegratorConfig object.
     */
    Integrator(Particle& particle, const IntegratorConfigDict& config);

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Integrator();

    /**
     * @brief Virtual method to perform a single integration step.
     */
    virtual void step() = 0;
};

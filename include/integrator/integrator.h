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
 * @brief Integrator class.
 * 
 * Base class for all integrators.
 * An integrator is responsible for updating the state of the system through some means.
 */
class Integrator {
protected:
    Particle& particle;  // Reference to Particle object

public:
    ConfigDict& config;  // Reference to IntegratorConfig object

    /**
     * @brief Constructor for Integrator class.
     * 
     * @param particle Reference to Particle object.
     * @param config Reference to IntegratorConfig object.
     */
    Integrator(Particle& particle, ConfigDict& config);

    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~Integrator();

    /**
     * @brief Virtual method to perform a single integration step.
     */
    virtual void step() = 0;
};

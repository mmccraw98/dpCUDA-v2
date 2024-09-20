#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "../particle/particle.h"

/**
 * @class Orchestrator
 * @brief Manages the orchestration of particle logging and calculations.
 *
 * The Orchestrator class is responsible for managing the logging of
 * various properties of a Particle object.
 * Importantly, it calculates all necessary values (if any) from which all
 * log variables are calculated - and does so in one step.  This ensures
 * that the values are calculated in a consistent manner and is done
 * nearly as efficiently as possible.
 * It additionally applies any necessary modifiers to the log variables.
 * For instance, x/N, x/Nv, x/dof normalizes quantity x by the number
 * of particles, vertices, or degrees of freedom in the system.
 */
class Orchestrator {
protected:
    Particle& particle;  // The particle object that the orchestrator is orchestrating
    std::vector<std::string> unmodified_log_names;  // Log variable names without any modifiers
    std::vector<std::string> modifiers = {"/"};

    /**
     * @brief Removes the possible modifier from a given log name
     * @param name The name to remove the modifier from
     * @return The unmodified name
     */
    std::string get_unmodified_log_name(const std::string& name);

public:
    /**
     * @brief Constructs an Orchestrator object.
     * @param particle The particle object to orchestrate.
     * @param log_names The names of the variables to log.
     */
    Orchestrator(Particle& particle, const std::vector<std::string>& log_names);

    /**
     * @brief Destructor for the Orchestrator object.
     */
    ~Orchestrator();

    std::vector<std::string> log_names;  // The name of the variables that are set to be logged

    /**
     * @brief Sets the log names and unmodified log names
     * @param log_names The names of the variables to be logged
     */
    void set_log_names(const std::vector<std::string>& log_names);

    /**
     * @brief Precalculates values needed for logging.
     */
    void precalculate();

    /**
     * @brief Gets the value of a log variable.
     * @param unmodified_log_name The unmodified name of the log variable.
     * @return The value of the log variable.
     */
    double get_value(const std::string& unmodified_log_name);

    /**
     * @brief Applies a modifier to a given value
     * @param name The name of the variable to apply the modifier to
     * @param value The value to apply the modifier to
     * @return The modified value
     */
    double apply_modifier(std::string& name, double value);
};

#endif /* ORCHESTRATOR_H */
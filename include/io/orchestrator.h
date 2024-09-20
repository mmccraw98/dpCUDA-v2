#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "../particle/particle.h"

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
    Orchestrator(Particle& particle, const std::vector<std::string>& log_names);
    ~Orchestrator();

    std::vector<std::string> log_names;  // The name of the variables that are set to be logged

    /**
     * @brief Sets the log names and unmodified log names
     * @param log_names The names of the variables to be logged
     */
    void set_log_names(const std::vector<std::string>& log_names);

    /**
     * @brief Precalculates certain values that are needed before they can be calculated
     * @details Certain values need things precalculated before they can be calculated (i.e. temperature needs kinetic energy) - multiple values may need the same thing precalculated, so this groups together precalculated quantities that are used for various log variables, it calculates the value for a given name
     */
    void precalculate();

    /**
     * @brief Gets the value of a given unmodified log name
     * @param unmodified_log_name The name of the variable to get the value of
     * @return The value of the variable
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
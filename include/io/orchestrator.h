#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "../particle/particle.h"
#include "utils.h"

class Orchestrator {
private:
    std::vector<std::shared_ptr<LogGroup>> log_groups; // Stores all LogGroups

public:
    Orchestrator(const std::vector<std::shared_ptr<LogGroup>>& log_groups) : log_groups(log_groups) {}

    std::unordered_set<std::string> get_unique_unmodified_log_names(long step) {
        std::unordered_set<std::string> unique_log_names;

        // Iterate through each log group and check if it's active for this step
        for (const auto& log_group : log_groups) {
            if (log_group->should_log(step)) {
                // Get the log names and insert the unmodified names into the set
                auto group_log_names = log_group->get_log_names();
                for (const auto& log_name : group_log_names) {
                    unique_log_names.insert(get_unmodified_log_name(log_name));
                }
            }
        }

        return unique_log_names; // This will contain only unique unmodified log names
    }

private:
    std::string get_unmodified_log_name(const std::string& name) {
        // Remove any modifiers like "/N", "/Nv", etc.
        size_t pos = name.find('/');
        return (pos != std::string::npos) ? name.substr(0, pos) : name;
    }
};


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

#endif /* ORCHESTRATOR_H */
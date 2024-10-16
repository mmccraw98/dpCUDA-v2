#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "../particle/particle.h"
#include "../integrator/integrator.h"
#include "utils.h"

/**
 * @brief Orchestrator class acts as an interface between the LogGroups and the simulation objects.
 * Manages pre-requisite calculations for log variables and provides values to the LogGroups.
 * 
 */
class Orchestrator {
private:
    Particle& particle;  // the particle object
    Integrator* integrator; // the integrator object
    bool has_integrator; // whether the integrator is present
    std::unordered_map<std::string, bool> pre_req_calculation_status; // whether the pre-requisite calculations have been done

public:
    /**
     * @brief Construct a new Orchestrator object
     * 
     * @param particle The particle object
     * @param integrator The integrator object
     */
    Orchestrator(Particle& particle, Integrator* integrator = nullptr);
    ~Orchestrator();

    /**
     * @brief Initializes the pre-requisite calculation status
     * Determines which pre-requisite calculations need to be done for each log name
     * 
     */
    void init_pre_req_calculation_status();

    /**
     * @brief Handles the pre-requisite calculations for a given log name
     * 
     */
    void handle_pre_req_calculations(const std::string& log_name);
    
    /**
     * @brief Applies a modifier to a given value
     * @param modifier The modifier to apply
     * @param value The value to apply the modifier to
     * @return The modified value
     */
    double apply_modifier(std::string& modifier, double value);

    /**
     * @brief Gets the size of a vector log variable
     * @param unmodified_log_name The unmodified name of the log variable
     * @return The size of the vector log variable
     */
    std::vector<long> get_vector_size(const std::string& unmodified_log_name);
    
    /**
     * @brief Gets the value of a log variable
     * @param unmodified_log_name The unmodified name of the log variable
     * @param step The current step
     * @return The value of the log variable
     */
    template <typename T>
    T get_value(const std::string& unmodified_log_name, long step) {
        if (pre_req_calculation_status.find(unmodified_log_name) != pre_req_calculation_status.end()) {
            handle_pre_req_calculations(unmodified_log_name);
        }

        if (unmodified_log_name == "step") {
            return step;
        } else if (unmodified_log_name == "KE") {
            return particle.totalKineticEnergy();
        } else if (unmodified_log_name == "PE") {
            return particle.totalPotentialEnergy();
        } else if (unmodified_log_name == "TE") {
            return particle.totalEnergy();
        } else if (unmodified_log_name == "T") {
            return particle.calculateTemperature();
        } else {
            std::cerr << "Orchestrator::get_value: Log name not recognized: " << unmodified_log_name << std::endl;
            return 0.0;
        }
    }
    
    /**
     * @brief Gets the value of a vector log variable
     * @param unmodified_log_name The unmodified name of the log variable
     * @return The value of the log variable
     */
    template <typename T>
    thrust::host_vector<T> get_vector_value(const std::string& unmodified_log_name) {
        return particle.getArray<T>("d_" + unmodified_log_name);
    }

    /**
     * @brief Gets the type of an array
     * @param array_name The name of the array
     * @return The type of the array
     */
    std::string get_array_type(const std::string& array_name) {
        return particle.getArrayType("d_" + array_name);
    }
};

#endif /* ORCHESTRATOR_H */
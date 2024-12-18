#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <set>
#include <unordered_set>
#include "../particles/base/particle.h"
#include "../integrator/integrator.h"
#include "utils.h"
#include "../include/data/array_data.h"

class Orchestrator {
private:
    Particle& particle;  // the particle object
    Integrator* integrator; // the integrator object
    bool has_integrator; // whether the integrator is present
    std::set<std::string> dependencies;

public:

    Orchestrator(Particle& particle, Integrator* integrator = nullptr);
    ~Orchestrator();

    void define_dependencies(std::vector<std::string> all_log_names);
    void reset_dependency_status();
    void handle_dependencies(std::string log_name);
    bool is_dependent(std::string log_name);

    double apply_modifier(std::string& modifier, double value);

    template <typename T>
    T get_value(const std::string& unmodified_log_name, long step) {
        if (unmodified_log_name == "step") {
            return step;
        } else if (unmodified_log_name == "KE") {  // TODO: move all of these to the particle class
            return particle.totalKineticEnergy();
        } else if (unmodified_log_name == "PE") {
            return particle.totalPotentialEnergy();
        } else if (unmodified_log_name == "TE") {
            return particle.totalEnergy();
        } else if (unmodified_log_name == "T") {
            return particle.calculateTemperature();
        } else if (unmodified_log_name == "phi") {
            return particle.getPackingFraction();
        } else {
            std::cerr << "Orchestrator::get_value: Log name not recognized: " << unmodified_log_name << std::endl;
            return 0.0;
        }
    }

    bool arrays_need_reordering = false;
    std::unordered_map<std::string, ArrayData> get_reorder_index_data();

    ArrayData get_array_data(const std::string& unmodified_log_name);

};

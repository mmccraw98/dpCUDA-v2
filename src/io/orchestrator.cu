#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/io/utils.h"
#include "../../include/io/orchestrator.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>


// when the step starts
// check if the log groups should log
// determine the values to log
// log the values






// rework orchestrator to be more general and something that is constructed from a list of log-objects (log name list and step manager)

// orchestrator should be constructed from a list of a list of strings - the log names
//


// call this once at the beginning to define the log names and unique log names

// then, call the precalculate function, passing the list of log names involved in the calculation

Orchestrator::Orchestrator(Particle& particle, const std::vector<std::vector<std::string>>& log_names_list) : particle(particle) {
    log_names = extract_unique_log_names(log_names_list);
    unmodified_log_names = extract_unique_unmodified_log_names(log_names);
}

Orchestrator::~Orchestrator() {
    log_names.clear();
    unmodified_log_names.clear();
}

void Orchestrator::orchestrate(std::vector<std::vector<std::string>> log_names_list) {
    std::vector<std::string> new_log_names;
    for (const auto& unmodified_log_name : unmodified_log_names) {
        // calculate kinetic energy when it is needed
        if (unmodified_log_name == "KE" || unmodified_log_name == "TE" || unmodified_log_name == "T") {
            particle.calculateKineticEnergy();
        }
        // TODO: implement others (probably geometric ones)
    }
}

std::vector<std::string> Orchestrator::extract_unique_log_names(const std::vector<std::vector<std::string>>& log_names_list) {
    std::set<std::string> unique_log_names;
    for (const auto& log_names : log_names_list) {
        unique_log_names.insert(log_names.begin(), log_names.end());
    }
    return std::vector<std::string>(unique_log_names.begin(), unique_log_names.end());
}

std::vector<std::string> Orchestrator::extract_unique_unmodified_log_names(const std::vector<std::string>& log_names) {
    std::set<std::string> unique_unmodified_log_names;
    for (const auto& log_name : log_names) {
        unique_unmodified_log_names.insert(get_unmodified_log_name(log_name));
    }
    return std::vector<std::string>(unique_unmodified_log_names.begin(), unique_unmodified_log_names.end());
}

bool Orchestrator::is_modified_log_name(const std::string& name) {
    return name.find('/') != std::string::npos;
}

std::string Orchestrator::get_unmodified_log_name(const std::string& name) {
    size_t pos = name.find('/');
    if (pos != std::string::npos) {
        return name.substr(0, pos);
    }
    return name;
}

// TODO: change this to work with any type - particularly, vectors and matrices
double Orchestrator::get_value(const std::string& name) {
    std::string temp_name = get_unmodified_log_name(name);  // define new saved variables here - make sure to add them to the precalculate function if needed
    if (temp_name == "KE") {
        return particle.totalKineticEnergy();
    } else if (temp_name == "PE") {
        return particle.totalPotentialEnergy();
    } else if (temp_name == "TE") {
        return particle.totalEnergy();
    } else if (temp_name == "T") {
        return particle.calculateTemperature();
    } else if (temp_name == "phi") {
        return particle.getPackingFraction();
    } else {
        std::cerr << "Orchestrator::get_value: Unknown name: " << temp_name << std::endl;
        exit(EXIT_FAILURE);
    }
}

double Orchestrator::apply_modifier(std::string& name, double value) {
    if (name.size() >= 2 && name.substr(name.size() - 2) == "/N") {
        value /= particle.n_particles;
    } else if (name.size() >= 3 && name.substr(name.size() - 3) == "/Nv") {
        value /= particle.n_vertices;
    } else if (name.size() >= 4 && name.substr(name.size() - 4) == "/dof") {
        value /= particle.n_dof;
    }
    return value;
}
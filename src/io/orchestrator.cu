#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particles/base/particle.h"
#include "../../include/io/io_utils.h"
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

// initialize orchestrator:
// get dependencies from particle
// for each log name, check if it has a dependency - make list of all dependencies

// log:
// reset dependency status
// if any dependencies, for each dependency, calculate the dependency
// proceed with log

Orchestrator::Orchestrator(Particle& particle, Integrator* integrator) : particle(particle), integrator(integrator) {
    has_integrator = integrator != nullptr;
    if (particle.using_cell_list) {
        arrays_need_reordering = true;
    }
}

Orchestrator::~Orchestrator() {
}

void Orchestrator::reset_dependency_status() {
    particle.reset_dependency_status();
}

void Orchestrator::handle_dependencies(std::string log_name) { // io manager will give us all the log names and we check if any have dependencies then we get the particle to calculate them
    // print out the dependency status
    particle.calculate_dependencies(log_name);
}

bool Orchestrator::is_dependent(std::string log_name) {
    return particle.unique_dependents.find(log_name) != particle.unique_dependents.end();
}

double Orchestrator::apply_modifier(std::string& modifier, double value) {
    if (modifier == "N") {
        return value / particle.n_particles;
    } else if (modifier == "Nv") {
        return value / particle.n_vertices;
    } else if (modifier == "dof") {
        return value / particle.n_dof;
    } else {
        std::cerr << "Orchestrator::apply_modifier: Modifier not recognized: " << modifier << std::endl;
        return value;
    }
}

std::unordered_map<std::string, ArrayData> Orchestrator::get_reorder_index_data() {
    std::unordered_map<std::string, ArrayData> reorder_index_data;
    for (const std::string& index_name : particle.get_reorder_arrays()) {
        reorder_index_data[index_name] = particle.getArrayData(index_name);
    }
    return reorder_index_data;
}

ArrayData Orchestrator::get_array_data(const std::string& unmodified_log_name) {
    return particle.getArrayData(unmodified_log_name);
}

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



Orchestrator::Orchestrator(Particle& particle, Integrator* integrator) : particle(particle), integrator(integrator) {
    init_pre_req_calculation_status();
    has_integrator = integrator != nullptr;

    std::cout << "Orchestrator::Orchestrator: Getting radii and neighbor list" << std::endl;
    auto radii = get_vector_value<double>("radii");
    auto neighbor_list = get_vector_value<long>("neighbor_list");

}

Orchestrator::~Orchestrator() {
    pre_req_calculation_status.clear();
}

void Orchestrator::init_pre_req_calculation_status() {
    for (const std::string& log_name : particle.pre_req_calculations) {
        pre_req_calculation_status[log_name] = false;  // Initialize all to false
    }
}

void Orchestrator::handle_pre_req_calculations(const std::string& log_name) {
    // some variables have pre-req calculations, they are defined here
    if (log_name == "KE" || log_name == "TE" || log_name == "T" || log_name == "kinetic_energy") {  // i.e. to get KE, TE, or T, we first need to determine the kinetic energy
        if (!pre_req_calculation_status["KE"] || !pre_req_calculation_status["kinetic_energy"]) {
            particle.calculateKineticEnergy();
            pre_req_calculation_status["KE"] = true;
            pre_req_calculation_status["kinetic_energy"] = true;
        }
    }
    // fill in others here...
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

std::vector<long> Orchestrator::get_vector_size(const std::string& unmodified_log_name) {
    std::vector<long> size;
    if (unmodified_log_name == "something complicated") {
        // d > 2 here
    } else if (unmodified_log_name == "positions_x" || unmodified_log_name == "positions_y" || unmodified_log_name == "velocities_x" || unmodified_log_name == "velocities_y" || unmodified_log_name == "forces_x" || unmodified_log_name == "forces_y" || unmodified_log_name == "potential_energy" || unmodified_log_name == "kinetic_energy") {
        size = {particle.n_particles, 1};
    } else if (unmodified_log_name == "radii" || unmodified_log_name == "masses") {
        size = {particle.n_particles, 1};
    } else if (unmodified_log_name == "cell_index" || unmodified_log_name == "sorted_cell_index" || unmodified_log_name == "particle_index" || unmodified_log_name == "cell_start" || unmodified_log_name == "num_neighbors" || unmodified_log_name == "neighbor_list") {
        thrust::host_vector<long> temp = particle.getArray<long>("d_" + unmodified_log_name);  // yeah i dont like this at all.  in fact, i dislike all of the io stuff
        size = {static_cast<long>(temp.size()), 1};
    } else if (unmodified_log_name == "box_size") {
        size = {N_DIM, 1};
    } else {
        std::cerr << "Orchestrator::get_vector_size: Unrecognized log name: " << unmodified_log_name << std::endl;
    }
    return size;
}

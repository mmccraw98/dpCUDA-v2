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
    if (log_name == "KE" || log_name == "TE" || log_name == "T") {  // i.e. to get KE, TE, or T, we first need to determine the kinetic energy
        if (!pre_req_calculation_status["KE"]) {
            particle.calculateKineticEnergy();
            pre_req_calculation_status["KE"] = true;
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
    } else if (unmodified_log_name == "positions" || unmodified_log_name == "velocities" || unmodified_log_name == "forces") {
        size = {particle.n_particles, N_DIM};  // n x d
    } else if (unmodified_log_name == "something to do with vertices") {
        // num_vertices x d
    } else {
        size = {particle.n_particles, 1};  // n x 1
    }
    return size;
}

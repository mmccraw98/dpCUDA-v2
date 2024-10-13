#include "../include/io/orchestrator.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/integrator.h"
#include "../include/integrator/nve.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
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

template <typename ParticleType, typename IntegratorType>
Orchestrator<ParticleType, IntegratorType>::Orchestrator(ParticleType& particle, IntegratorType* integrator) : particle(particle), integrator(integrator) {
    this->has_integrator = integrator != nullptr;
    this->init_pre_req_calculation_status();
}

template <typename ParticleType, typename IntegratorType>
Orchestrator<ParticleType, IntegratorType>::~Orchestrator() {
    this->pre_req_calculation_status.clear();
}

template <typename ParticleType, typename IntegratorType>
void Orchestrator<ParticleType, IntegratorType>::init_pre_req_calculation_status() {
    for (const std::string& log_name : this->particle.pre_req_calculations) {
        this->pre_req_calculation_status[log_name] = false;  // Initialize all to false
    }
}

template <typename ParticleType, typename IntegratorType>
void Orchestrator<ParticleType, IntegratorType>::handle_pre_req_calculations(const std::string& log_name) {
    if (log_name == "KE" || log_name == "TE" || log_name == "T") {
        if (!this->pre_req_calculation_status["KE"]) {
            this->particle.calculateKineticEnergy();
            this->pre_req_calculation_status["KE"] = true;
        }
    }
    // fill in others here...
}

template <typename ParticleType, typename IntegratorType>
double Orchestrator<ParticleType, IntegratorType>::apply_modifier(std::string& modifier, double value) {
    if (modifier == "N") {
        return value / this->particle.n_particles;
    } else if (modifier == "Nv") {
        return value / this->particle.n_vertices;
    } else if (modifier == "dof") {
        return value / this->particle.n_dof;
    } else {
        std::cerr << "Orchestrator::apply_modifier: Modifier not recognized: " << modifier << std::endl;
        return value;
    }
}

template <typename ParticleType, typename IntegratorType>
std::vector<long> Orchestrator<ParticleType, IntegratorType>::get_vector_size(const std::string& unmodified_log_name) {
    std::vector<long> size;
    if (unmodified_log_name == "something complicated") {
        // d > 2 here
    } else if (unmodified_log_name == "positions" || unmodified_log_name == "velocities" || unmodified_log_name == "forces") {
        size = {this->particle.n_particles, N_DIM};  // n x d
    } else if (unmodified_log_name == "something to do with vertices") {
        // num_vertices x d
    } else {
        size = {this->particle.n_particles, 1};  // n x 1
    }
    return size;
}

template <typename ParticleType, typename IntegratorType>
template <typename T>
T Orchestrator<ParticleType, IntegratorType>::get_value(const std::string& unmodified_log_name, long step) {
    if (this->pre_req_calculation_status.find(unmodified_log_name) != this->pre_req_calculation_status.end()) {
        this->handle_pre_req_calculations(unmodified_log_name);
    }

    if (unmodified_log_name == "step") {
        return step;
    } else if (unmodified_log_name == "KE") {
        return this->particle.totalKineticEnergy();
    } else if (unmodified_log_name == "PE") {
        return this->particle.totalPotentialEnergy();
    } else if (unmodified_log_name == "TE") {
        return this->particle.totalEnergy();
    } else if (unmodified_log_name == "T") {
        return this->particle.calculateTemperature();
    } else {
        std::cerr << "Orchestrator::get_value: Log name not recognized: " << unmodified_log_name << std::endl;
        return 0.0;
    }
}

template <typename ParticleType, typename IntegratorType>
template <typename T>
thrust::host_vector<T> Orchestrator<ParticleType, IntegratorType>::get_vector_value(const std::string& unmodified_log_name) {
    return this->particle.template getArray<T>(unmodified_log_name);
}



template class Orchestrator<Disk, NVE<Disk>>;
template double Orchestrator<Disk, NVE<Disk>>::get_value<double>(const std::string& unmodified_log_name, long step);
template thrust::host_vector<double> Orchestrator<Disk, NVE<Disk>>::get_vector_value<double>(const std::string& unmodified_log_name);
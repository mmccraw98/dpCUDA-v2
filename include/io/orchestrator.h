#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
#include "../../include/integrator/integrator.h"
#include "../../include/integrator/nve.h"
#include "utils.h"
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

// since orchestrator is now templated, everything needs to be defined in the header

template <typename ParticleType, typename IntegratorType>
class Orchestrator {
private:
    ParticleType& particle;  // the particle object (and integrator if needed) are private since the orchestrator handles everything
    IntegratorType* integrator;
    bool has_integrator;
    std::unordered_map<std::string, bool> pre_req_calculation_status;

public:
    Orchestrator(ParticleType& particle, IntegratorType* integrator = nullptr);  // can pass the integrator here too - this way, the log groups dont need to know about where the values are coming from
    ~Orchestrator();

    void init_pre_req_calculation_status();

    void handle_pre_req_calculations(const std::string& log_name);

    double apply_modifier(std::string& modifier, double value);

    std::vector<long> get_vector_size(const std::string& unmodified_log_name);
    
    template <typename T>
    T get_value(const std::string& unmodified_log_name, long step);
    
    template <typename T>
    thrust::host_vector<T> get_vector_value(const std::string& unmodified_log_name);
};


#endif /* ORCHESTRATOR_H */
#ifndef STATE_LOG_IMPL_H
#define STATE_LOG_IMPL_H

#include "state_log.h"

template <typename ParticleType, typename IntegratorType>
StateLog<ParticleType, IntegratorType>::StateLog(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix, const std::string& extension)
    : BaseLogGroup<ParticleType, IntegratorType>(config, orchestrator), root_path(root_path), indexed_file_prefix(indexed_file_prefix), extension(extension) {  // Fix: Explicitly initialize the base class with template arguments
}

template <typename ParticleType, typename IntegratorType>
StateLog<ParticleType, IntegratorType>::~StateLog() {
}

template <typename ParticleType, typename IntegratorType>
void StateLog<ParticleType, IntegratorType>::log(long step) {
    for (int i = 0; i < this->config.log_names.size(); i++) {
        // Fix: Specify 'template' before get_vector_value
        thrust::host_vector<double> value = this->orchestrator.template get_vector_value<double>(this->config.log_names[i]);
        std::vector<long> size = this->orchestrator.get_vector_size(this->config.log_names[i]);

        std::filesystem::path file_path = std::filesystem::path(this->root_path) / (this->indexed_file_prefix + std::to_string(step)) / (this->config.log_names[i] + this->extension);

        write_array_to_file(
            file_path.string(),
            value,
            size[0],
            size[1],
            this->precision
        );
    }
}

#endif /* STATE_LOG_IMPL_H */

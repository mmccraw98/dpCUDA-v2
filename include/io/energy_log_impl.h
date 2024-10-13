#ifndef ENERGY_LOG_IMPL_H
#define ENERGY_LOG_IMPL_H

#include "energy_log.h"

template <typename ParticleType, typename IntegratorType>
EnergyLog<ParticleType, IntegratorType>::EnergyLog(LogGroupConfig log_group_config, Orchestrator<ParticleType, IntegratorType>& orchestrator, const std::string& file_name, bool overwrite)
    : MacroLog<ParticleType, IntegratorType>(log_group_config, orchestrator) {
    this->file_name = file_name;
    this->log_file = open_output_file(file_name, overwrite);
    if (!this->log_file.is_open()) {
        std::cerr << "ERROR: EnergyLog: could not open file: " << this->file_name << std::endl;
        exit(1);
    }
    if (this->log_file.tellp() != 0) {
        this->has_header = true;
    } else {
        write_header();
    }
}

template <typename ParticleType, typename IntegratorType>
EnergyLog<ParticleType, IntegratorType>::~EnergyLog() {
    if (this->log_file.is_open()) {
        this->log_file.close();
    }
}

template <typename ParticleType, typename IntegratorType>
void EnergyLog<ParticleType, IntegratorType>::write_header() {
    for (size_t i = 0; i < this->config.log_names.size(); ++i) {
        this->log_file << this->config.log_names[i];
        if (i < this->config.log_names.size() - 1) {
            this->log_file << delimiter;
        }
    }
    this->log_file << "\n";
    this->log_file.flush();
}

template <typename ParticleType, typename IntegratorType>
void EnergyLog<ParticleType, IntegratorType>::log(long step) {
    for (size_t i = 0; i < this->config.log_names.size(); ++i) {
        double value = this->orchestrator.template get_value<double>(this->unmodified_log_names[i], step);
        if (this->log_name_is_modified(this->config.log_names[i])) {
            std::string modifier = this->get_modifier(this->config.log_names[i]);
            value = this->orchestrator.apply_modifier(modifier, value);
        }
        this->log_file << std::fixed << std::setprecision(this->precision) << value;
        if (i < this->config.log_names.size() - 1) {
            this->log_file << delimiter;
        }
    }
    this->log_file << "\n";
    this->log_file.flush();
}



#endif /* ENERGY_LOG_IMPL_H */
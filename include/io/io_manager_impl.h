#ifndef IO_MANAGER_IMPL_H
#define IO_MANAGER_IMPL_H


#include "io_manager.h"


template <typename ParticleType, typename IntegratorType>
IOManager<ParticleType, IntegratorType>::IOManager(ParticleType& particle, IntegratorType* integrator, std::vector<LogGroupConfig> log_configs, std::string root_path, bool overwrite) : particle(particle), integrator(integrator), orchestrator(particle, integrator), root_path(root_path), overwrite(overwrite), log_configs(log_configs) {
    // probably validate root_path if it is not empty

    for (auto& config : this->log_configs) {

        if (config.group_name == "energy") {
            if (this->system_dir_path.empty()) {
                init_path(this->system_dir_path, this->system_dir_name);
                make_dir(this->system_dir_path, overwrite);  // may need to change function signature
            }
            std::filesystem::path energy_file_path = this->system_dir_path / (this->energy_file_name + this->energy_file_extension);
            this->log_groups.push_back(std::make_unique<EnergyLog<ParticleType, IntegratorType>>(config, orchestrator, energy_file_path, overwrite));

        } else if (config.group_name == "console") {
            this->log_groups.push_back(std::make_unique<ConsoleLog<ParticleType, IntegratorType>>(config, orchestrator));

        } else if (config.group_name == "state") {
            if (this->trajectory_dir_path.empty()) {
                init_path(this->trajectory_dir_path, this->trajectory_dir_name);
                make_dir(this->trajectory_dir_path, overwrite);  // may need to change function signature
            }
            this->log_groups.push_back(std::make_unique<StateLog<ParticleType, IntegratorType>>(config, orchestrator, this->trajectory_dir_path, this->indexed_file_prefix, this->state_file_extension));
        }
    }

    long num_log_groups = this->log_groups.size();
    for (long i = 0; i < num_log_groups; i++) {
        // std::cout << log_groups[i]->config.to_json().dump(4) << std::endl;
        std::cout << this->log_groups[i]->config.group_name << std::endl;
    }
}

template <typename ParticleType, typename IntegratorType>
IOManager<ParticleType, IntegratorType>::~IOManager() {
}

template <typename ParticleType, typename IntegratorType>
void IOManager<ParticleType, IntegratorType>::init_path(std::filesystem::path& path, const std::string& path_name) {
    if (this->root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_path:" << path_name << " root_path is empty" << std::endl;
        return;
    }
    path = static_cast<std::filesystem::path>(this->root_path) / static_cast<std::filesystem::path>(path_name);
}

template <typename ParticleType, typename IntegratorType>
void IOManager<ParticleType, IntegratorType>::log(long step) {
    bool log_required = false;
    for (const auto& log_group : log_groups) {
        log_group->update_log_status(step);
        if (log_group->should_log) {
            log_required = true;
        }
    }

    if (log_required) {
        this->orchestrator.init_pre_req_calculation_status();
        for (const auto& log_group : log_groups) {
            if (log_group->should_log) {
                log_group->log(step);
            }
        }
    }
}

template <typename ParticleType, typename IntegratorType>
void IOManager<ParticleType, IntegratorType>::write_log_configs(std::filesystem::path path) {
    for (auto& config : this->log_configs) {
        std::cout << config.to_json().dump(4) << std::endl;
    }
}

template <typename ParticleType, typename IntegratorType>
void IOManager<ParticleType, IntegratorType>::write_particle_config(std::filesystem::path path) {
    // std::cout << particle.config.to_json().dump(4) << std::endl;
}

#endif /* IO_MANAGER_IMPL_H */
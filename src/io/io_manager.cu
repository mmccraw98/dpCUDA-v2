#include "../../include/io/io_manager.h"
#include <iostream>
#include <filesystem>

IOManager::IOManager(Particle& particle, Integrator& integrator, std::vector<LogGroupConfig> log_configs, std::string root_path, bool overwrite) : particle(particle), integrator(integrator), orchestrator(particle, &integrator), root_path(root_path), overwrite(overwrite), log_configs(log_configs) {
    // probably validate root_path if it is not empty

    for (auto& config : log_configs) {

        if (config.group_name == "energy") {
            if (system_dir_path.empty()) {
                init_path(system_dir_path, system_dir_name);
                make_dir(system_dir_path, overwrite);  // may need to change function signature
            }
            std::filesystem::path energy_file_path = system_dir_path / (energy_file_name + energy_file_extension);
            log_groups.push_back(new EnergyLog(config, orchestrator, energy_file_path, overwrite));

        } else if (config.group_name == "console") {
            log_groups.push_back(new ConsoleLog(config, orchestrator));

        } else if (config.group_name == "state") {
            if (trajectory_dir_path.empty()) {
                init_path(trajectory_dir_path, trajectory_dir_name);
                make_dir(trajectory_dir_path, overwrite);  // may need to change function signature
            }
            log_groups.push_back(new StateLog(config, orchestrator, trajectory_dir_path, indexed_file_prefix, state_file_extension));
        }
    }

    

    long num_log_groups = log_groups.size();
    for (long i = 0; i < num_log_groups; i++) {
        // std::cout << log_groups[i]->config.to_json().dump(4) << std::endl;
        std::cout << log_groups[i]->config.group_name << std::endl;
    }
}

IOManager::~IOManager() {
    for (auto& log_group : log_groups) {
        delete log_group;
    }
}

void IOManager::init_path(std::filesystem::path& path, const std::string& path_name) {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_path:" << path_name << " root_path is empty" << std::endl;
        return;
    }
    path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(path_name);
}

void IOManager::log(long step) {
    bool log_required = false;
    for (BaseLogGroup* log_group : log_groups) {
        log_group->update_log_status(step);
        if (log_group->should_log) {
            log_required = true;
        }
    }

    if (log_required) {
        orchestrator.init_pre_req_calculation_status();
        for (BaseLogGroup* log_group : log_groups) {
            if (log_group->should_log) {
                log_group->log(step);
            }
        }
    }
}

void IOManager::write_log_configs(std::filesystem::path path) {
    for (auto& config : log_configs) {
        std::cout << config.to_json().dump(4) << std::endl;
    }
}

void IOManager::write_particle_config(std::filesystem::path path) {
    // std::cout << particle.config.to_json().dump(4) << std::endl;
}
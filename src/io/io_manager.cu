#include "../../include/io/io_manager.h"
#include <iostream>
#include <filesystem>

IOManager::IOManager(Particle& particle, Integrator& integrator, std::vector<LogGroupConfig> log_configs, std::string root_path, bool overwrite) : particle(particle), integrator(integrator), orchestrator(particle, &integrator), root_path(root_path), overwrite(overwrite) {
    // probably validate root_path if it is not empty
    // set up system path if root is not empty
    // set up trajectory path if root is not empty and saving trajectories is enabled
    for (auto& config : log_configs) {
        if (config.group_name == "energy") {
            if (system_dir_path.empty()) {
                init_system_path();
            }
            std::filesystem::path energy_file_path = system_dir_path / (energy_file_name + energy_file_extension);
            make_dir(energy_file_path.parent_path(), overwrite);
            log_groups.push_back(new EnergyLog(config, orchestrator, energy_file_path, overwrite));
        } else if (config.group_name == "console") {
            log_groups.push_back(new ConsoleLog(config, orchestrator));
        }
        // check if saving trajectories is enabled
        // check if saving parameters is enabled
    }
}

IOManager::~IOManager() {
    for (auto& log_group : log_groups) {
        delete log_group;
    }
}

void IOManager::init_system_path() {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_system_path: root_path is empty" << std::endl;
        return;
    }
    system_dir_path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(system_dir_name);
}

void IOManager::init_trajectory_path() {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_trajectory_path: root_path is empty" << std::endl;
        return;
    }
    trajectory_dir_path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(trajectory_dir_name);
}

void IOManager::init_restart_path() {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_restart_path: root_path is empty" << std::endl;
        return;
    }
    restart_dir_path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(restart_dir_name);
}

void IOManager::init_init_path() {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_init_path: root_path is empty" << std::endl;
        return;
    }
    init_dir_path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(init_dir_name);
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

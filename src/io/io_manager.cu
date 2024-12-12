#include "../../include/io/io_manager.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <vector>

IOManager::IOManager(std::vector<LogGroupConfig> log_configs, Particle& particle, Integrator* integrator, std::string root_path, long num_threads, bool overwrite) : particle(particle), integrator(integrator), orchestrator(particle, integrator), root_path(root_path), num_threads(num_threads), overwrite(overwrite), log_configs(log_configs), thread_pool(num_threads) {
    // probably validate root_path if it is not empty
    use_parallel = num_threads > 1;

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
        
        } else if (config.group_name == "init") {
            if (system_dir_path.empty()) {
                init_path(system_dir_path, system_dir_name);
                make_dir(system_dir_path, overwrite);  // may need to change function signature
            }
            state_log = new StateLog(config, orchestrator, system_dir_path, "", state_file_extension);

        } else {
            std::cerr << "ERROR: IOManager::IOManager:" << config.group_name << " is not a valid log group name" << std::endl;
        }
    }

    // define the dependencies in the log groups
    for (auto& log_group : log_groups) {
        log_group->define_dependencies();
    }

    if (state_log != nullptr) {
        // make the init and restart directories
        init_dir_path = system_dir_path / init_dir_name;
        make_dir(init_dir_path.string(), true);
        restart_dir_path = system_dir_path / restart_dir_name;
        make_dir(restart_dir_path.string(), true);

        // gather the data
        state_log->gather_data(0);
        // write the data
        state_log->write_state_to_path(init_dir_path);
    }
}

IOManager::~IOManager() {
    thread_pool.shutdown();
    for (auto& log_group : log_groups) {
        delete log_group;
    }
}

void IOManager::write_restart_file(long step) {
    // write the state to the restart directory
    state_log->write_state_to_path(restart_dir_path);
    // write the current step to a file using the write_json_to_file function
    write_json_to_file(restart_dir_path / "current_step.json", nlohmann::json{{"step", step}});
}

void IOManager::init_path(std::filesystem::path& path, const std::string& path_name) {
    if (root_path.empty()) {
        std::cerr << "ERROR: IOManager::init_path:" << path_name << " root_path is empty" << std::endl;
        return;
    }
    path = static_cast<std::filesystem::path>(root_path) / static_cast<std::filesystem::path>(path_name);
}

void IOManager::log(long step, bool force) {
    // figure out if any logs need to be written
    bool log_required = false;
    for (BaseLogGroup* log_group : log_groups) {
        log_group->update_log_status(step);
        if (log_group->should_log || force) {
            log_required = true;
        }
    }

    // do the logging

    if (log_required) {
        // handle dependency calculation if any
        orchestrator.reset_dependency_status();
        for (BaseLogGroup* log_group : log_groups) {
            if (log_group->has_dependencies) {
                log_group->handle_dependencies();
            }
        }

        // gather the data
        for (BaseLogGroup* log_group : log_groups) {
            if (log_group->should_log || force) {
                log_group->gather_data(step);
            }
        }

        // now we can disconnect from the simulation and run these in parallel

        // log
        std::vector<std::thread> threads;  // Store threads for async log groups
        for (BaseLogGroup* log_group : log_groups) {
            if (log_group->should_log || force) {
                if (log_group->parallel && use_parallel) {
                    thread_pool.enqueue([log_group, step]() {
                        log_group->log(step);
                    });
                } else {
                    log_group->log(step);
                }
            }
        }

        // Detach all threads
        for (auto& thread : threads) {
            thread.detach();  // Let them run independently
        }
    }
}

void IOManager::write_log_configs(std::filesystem::path path) {
    nlohmann::json all_configs_json;
    for (auto& config : log_configs) {
        all_configs_json[config.group_name] = config.to_json();
    }
    write_json_to_file(path / "log_configs.json", all_configs_json);
}

void IOManager::write_particle_config(std::filesystem::path path) {
    write_json_to_file(path / "particle_config.json", particle.config->to_json());
}

void IOManager::write_integrator_config(std::filesystem::path path) {
    write_json_to_file(path / "integrator_config.json", integrator->config.to_json());
}

void IOManager::write_params() {
    if (system_dir_path.empty()) {
        init_path(system_dir_path, system_dir_name);
        make_dir(system_dir_path, overwrite);  // may need to change function signature
    }
    write_log_configs(system_dir_path);
    write_particle_config(system_dir_path);
    if (integrator != nullptr) {
        write_integrator_config(system_dir_path);
    }
    // TODO: write run params
}

// void IOManager::write_state_to_path() {
//     if (state_log == nullptr) {
//         std::cerr << "ERROR: IOManager::write_state_to_path: state_log is not initialized" << std::endl;
//         return;
//     }
//     state_log->write_state_to_path();
// }
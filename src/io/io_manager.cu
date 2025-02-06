#include "../../include/io/io_manager.h"
#include <iostream>
#include <filesystem>
#include <thread>
#include <vector>

IOManager::IOManager(std::vector<ConfigDict> log_configs,
                     Particle& particle,
                     Integrator* integrator,
                     std::string root,
                     long num_threads,
                     bool overwrite)
    : particle(particle),
      integrator(integrator),
      orchestrator(particle, integrator),
      log_configs(log_configs),
      overwrite(overwrite),
      num_threads(num_threads),
      root(root),
      // The thread pool is initialized last in initializer list 
      thread_pool(num_threads)
{
    // Root path creation
    root_path = std::filesystem::path(root);
    if (!root_path.empty()) {
        if (overwrite) {
            std::filesystem::remove_all(root_path);
            std::filesystem::create_directories(root_path);
        } else if (std::filesystem::exists(root_path)) {
            std::cerr << "ERROR: IOManager::IOManager: root path " << root_path
                      << " already exists and overwriting is disabled!\n";
            return;
        }
    }

    use_parallel = (num_threads > 1);

    // Loop through each log config and create relevant log groups
    for (auto& config : log_configs) {
        const std::string group_name = config.at("group_name").get<std::string>();

        if (group_name == "energy") {
            // Ensure system_dir_path is set up
            if (system_dir_path.empty()) {
                init_path(system_dir_path, system_dir_name);
                make_dir(system_dir_path, overwrite);
            }
            // Assign to the *member* variable energy_file_path
            energy_file_path = system_dir_path / (energy_file_name + energy_file_extension);
            // Then create the log group
            log_groups.push_back(new EnergyLog(config, orchestrator, energy_file_path, overwrite));

        } else if (group_name == "console") {
            log_groups.push_back(new ConsoleLog(config, orchestrator));

        } else if (group_name == "state") {
            if (trajectory_dir_path.empty()) {
                init_path(trajectory_dir_path, trajectory_dir_name);
                make_dir(trajectory_dir_path, overwrite);
            }
            log_groups.push_back(
                new StateLog(config, orchestrator,
                             trajectory_dir_path,
                             indexed_file_prefix,
                             state_file_extension)
            );

        } else if (group_name == "restart") {
            if (system_dir_path.empty()) {
                init_path(system_dir_path, system_dir_name);
                make_dir(system_dir_path, overwrite);
            }
            // Create a special StateLog for restarts
            state_log = new StateLog(config, orchestrator,
                                     system_dir_path,
                                     /*prefix=*/"",
                                     state_file_extension,
                                     /*restart_mode=*/true);
            log_groups.push_back(state_log);

        } else {
            std::cerr << "ERROR: IOManager::IOManager: " << group_name
                      << " is not a valid log group name\n";
        }
    }

    // Define dependencies in the log groups
    for (auto& log_group : log_groups) {
        log_group->define_dependencies();
    }

    // If we have a "restart" group, also write the initial state
    if (state_log != nullptr) {
        init_dir_path = system_dir_path / init_dir_name;
        make_dir(init_dir_path.string(), true);
        restart_dir_path = system_dir_path / restart_dir_name;
        make_dir(restart_dir_path.string(), true);

        state_log->gather_data(0);
        state_log->write_state_to_path(init_dir_path);
    }
}

IOManager::~IOManager() {
    // Ensure all queued tasks complete before deleting log groups
    thread_pool.shutdown();

    for (auto& log_group : log_groups) {
        delete log_group;
    }
}

void IOManager::init_path(std::filesystem::path& path, const std::string& path_name) {
    if (root.empty()) {
        std::cerr << "ERROR: IOManager::init_path: root is empty. path_name="
                  << path_name << std::endl;
        return;
    }
    path = std::filesystem::path(root) / std::filesystem::path(path_name);
}

void IOManager::log(long step, bool force) {
    bool log_required = false;
    for (BaseLogGroup* log_group : log_groups) {
        log_group->update_log_status(step);
        if (log_group->should_log || force) {
            log_required = true;
        }
    }
    if (!log_required) return;

    // Handle dependencies
    orchestrator.reset_dependency_status();
    for (BaseLogGroup* log_group : log_groups) {
        if (log_group->has_dependencies) {
            log_group->handle_dependencies();
        }
    }

    // Gather data
    for (BaseLogGroup* log_group : log_groups) {
        if (log_group->should_log || force) {
            log_group->gather_data(step);
        }
    }

    // Enqueue or run the actual logging
    for (BaseLogGroup* log_group : log_groups) {
        if (log_group->should_log || force) {
            if (log_group->parallel && use_parallel) {
                // Create a snapshot and wrap it in a shared_ptr
                auto snapshot = log_group->snapshot();
                auto local_log_group = std::shared_ptr<BaseLogGroup>(std::move(snapshot));
                // Enqueue a lambda that captures the shared_ptr (which is copyable)
                thread_pool.enqueue([local_log_group, step]() {
                    local_log_group->log(step);
                });
            } else {
                log_group->log(step);
            }
        }
    }
}

void IOManager::write_restart_file(long step) {
    if (!state_log) return;
    // Write the state to the restart directory
    state_log->write_state_to_path(restart_dir_path);

    // Write the current step
    write_json_to_file(restart_dir_path / "current_step.json",
                       nlohmann::json{{"step", step}});
}

void IOManager::write_log_configs(std::filesystem::path path) {
    for (auto& config : log_configs) {
        std::string group_name = config.at("group_name").get<std::string>();
        config.save(path / (group_name + "_log_config.json"));
    }
}

void IOManager::write_particle_config(std::filesystem::path path) {
    particle.config.save(path / "particle_config.json");
}

void IOManager::write_integrator_config(std::filesystem::path path) {
    if (!integrator) return;
    integrator->config.save(path / "integrator_config.json");
}

void IOManager::write_params() {
    if (system_dir_path.empty()) {
        init_path(system_dir_path, system_dir_name);
        make_dir(system_dir_path, overwrite);
    }
    write_log_configs(system_dir_path);
    write_particle_config(system_dir_path);
    if (integrator != nullptr) {
        write_integrator_config(system_dir_path);
    }
    // TODO: write integrator config to the root directory
}
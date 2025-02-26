#pragma once

#include "../particles/base/particle.h"
#include "../integrator/integrator.h"
#include "../../include/utils/thread_pool.h"

#include "io_utils.h"
#include "orchestrator.h"
#include "base_log_groups.h"
#include "energy_log.h"
#include "console_log.h"
#include "state_log.h"
#include <iostream>
#include <filesystem>

/**
 * @brief Manages all input and output operations
 */
class IOManager {
public:
    /**
     * @brief Constructor for IOManager
     * @param log_configs The configuration for each log group
     * @param particle The particle object
     * @param integrator The integrator object
     * @param root The root path for all output files
     * @param num_threads Number of threads for parallel IO
     * @param overwrite Whether to overwrite existing files
     */
    IOManager(std::vector<ConfigDict> log_configs,
              Particle& particle,
              Integrator* integrator = nullptr,
              std::string root = "",
              long num_threads = 1,
              bool overwrite = false);

    /**
     * @brief Destructor. Ensures thread pool is shut down before cleaning up logs.
     */
    ~IOManager();

    /**
     * @brief Logs the current state of the system using all log groups
     * @param step The current step of the simulation
     * @param force Whether to force the logging even if no logs are required
     */
    void log(long step, bool force = false);

    /**
     * @brief Writes the parameters of the system to a file
     */
    void write_params();

    /**
     * @brief Writes the log configurations to a file
     * @param path The path to write the log configurations to
     */
    void write_log_configs(std::filesystem::path path);

    /**
     * @brief Writes the particle configurations to a file
     * @param path The path to write the particle configurations to
     */
    void write_particle_config(std::filesystem::path path);
    
    /**
     * @brief Writes the integrator configurations to a file
     * @param path The path to write the integrator configurations to
     */
    void write_integrator_config(std::filesystem::path path);

    /**
     * @brief Writes a restart file containing current state and step
     * @param step The current simulation step
     */
    void write_restart_file(long step, std::string dir_name);

private:
    // --- Main references and orchestrator
    Particle& particle;
    Integrator* integrator;
    Orchestrator orchestrator;

    // --- Logging groups and config
    std::vector<BaseLogGroup*> log_groups;
    std::vector<ConfigDict> log_configs;
    StateLog* state_log = nullptr;

    // --- Settings
    bool overwrite;
    bool use_parallel;
    long num_threads;
    std::string root;

    // --- File naming
    std::string energy_file_extension = ".csv";
    std::string state_file_extension  = ".dat";
    std::string indexed_file_prefix   = "t";
    std::string energy_file_name      = "energy";
    std::string system_dir_name       = "system";
    std::string trajectory_dir_name   = "trajectories";
    std::string restart_dir_name      = "restart";
    std::string init_dir_name         = "init";

    // --- Paths
    std::filesystem::path root_path;
    std::filesystem::path energy_file_path;
    std::filesystem::path system_dir_path;
    std::filesystem::path trajectory_dir_path;
    std::filesystem::path restart_dir_path;
    std::filesystem::path init_dir_path;

    // --- Thread pool for parallel logging tasks
    ThreadPool thread_pool; 

    /**
     * @brief Helper to set up a path under the root directory
     * @param path Reference to a `std::filesystem::path` to fill
     * @param path_name Sub-directory name
     */
    void init_path(std::filesystem::path& path, const std::string& path_name);
};
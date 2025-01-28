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
     * @param overwrite Whether to overwrite existing files
     */
    IOManager(std::vector<LogGroupConfigDict> log_configs, Particle& particle, Integrator* integrator = nullptr, std::string root = "", long num_threads = 1, bool overwrite = false);
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

    void write_restart_file(long step);

    // /**
    //  * @brief Writes the current state of the system to a file
    //  */
    // void write_state_to_path();

private:
    ThreadPool thread_pool;  // thread pool for parallel IO - declared first to ensure destruction order
    Particle& particle;  // particle object
    Integrator* integrator;  // integrator object
    Orchestrator orchestrator;  // orchestrator object
    std::vector<BaseLogGroup*> log_groups;  // log groups
    std::vector<LogGroupConfigDict> log_configs;  // log configurations
    StateLog* state_log = nullptr;  // state log object

    bool overwrite;  // whether to overwrite existing files
    bool use_parallel;  // whether to use parallel IO
    long num_threads;  // number of threads to use for parallel IO

    std::string root;  // root path for all output files
    
    std::string energy_file_extension = ".csv";//".csv";  // file extension for energy files
    std::string state_file_extension = ".dat";//".txt";  // file extension for state files
    std::string indexed_file_prefix = "t";  // indexed file prefix - trajectory/t{step}/state
    std::string energy_file_name = "energy";  // file name for energy files
    std::string system_dir_name = "system";  // directory name for system files
    std::string trajectory_dir_name = "trajectories";  // directory name for trajectory files
    std::string restart_dir_name = "restart";  // saves what is needed to restart, continuously overwrite with any updates
    std::string init_dir_name = "init";  // saves the initial configuration
    
    std::filesystem::path root_path;  // path to root directory
    std::filesystem::path energy_file_path;  // path to energy file
    std::filesystem::path system_dir_path;  // path to system directory
    std::filesystem::path trajectory_dir_path;  // path to trajectory directory
    std::filesystem::path restart_dir_path;  // path to restart directory
    std::filesystem::path init_dir_path;  // path to init directory

    /**
     * @brief Initializes the path for a directory
     * @param path The path to initialize
     * @param path_name The name of the path
     */
    void init_path(std::filesystem::path& path, const std::string& path_name);
};

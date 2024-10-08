#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "../particle/particle.h"
#include "../integrator/integrator.h"
#include "utils.h"
#include "orchestrator.h"
#include "base_log_groups.h"
#include "energy_log.h"
#include "console_log.h"
#include "state_log.h"
#include <iostream>
#include <filesystem>

class IOManager {
public:  // TODO: pass a config
    IOManager(Particle& particle, Integrator& integrator, std::vector<LogGroupConfig> log_configs, std::string root_path = "", bool overwrite = true);
    ~IOManager();
    void log(long step);


private:
    Particle& particle;
    Integrator& integrator;
    Orchestrator orchestrator;
    std::vector<BaseLogGroup*> log_groups;

    bool overwrite;

    std::string root_path;
    
    std::string energy_file_extension = ".csv";
    std::string state_file_extension = ".txt";
    std::string indexed_file_prefix = "t";
    std::string energy_file_name = "energy";
    std::string system_dir_name = "system";
    std::string trajectory_dir_name = "trajectories";
    std::string restart_dir_name = "restart";  // saves what is needed to restart, continuously overwrite with any updates
    std::string init_dir_name = "init";  // saves the initial configuration
    
    // TODO: these will probably be defined in some config if resuming a run
    std::filesystem::path energy_file_path;
    std::filesystem::path system_dir_path;
    std::filesystem::path trajectory_dir_path;
    std::filesystem::path restart_dir_path;
    std::filesystem::path init_dir_path;

    void init_path(std::filesystem::path& path, const std::string& path_name);

    void write_params(std::filesystem::path);
};

#endif /* IO_MANAGER_H */
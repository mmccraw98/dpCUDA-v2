#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "orchestrator.h"
#include "step_manager.h"
#include "../particle/particle.h"

 // TODO: move all non-file related functionality to iomanager

std::ifstream open_input_file(std::string file_name);
std::ofstream open_output_file(std::string file_name);
void make_dir(const std::string& dir_name, bool warn = true);
bool contains_substrings(const std::string& string, const std::vector<std::string>& substrings);
long get_largest_file_index(std::string dir_name, std::string file_prefix = "");

// make a python resume script that loads the configuration, gets the script details and arguments, and calls the relevant script with optionally overwriting some arguments

struct FileManagerConfig {
    std::string dir_name = "";
    std::vector<std::string> energy_log_names = {};
    std::vector<std::string> trajectory_log_names = {};
    std::vector<std::vector<std::string>> log_names_by_group = {};

    long energy_save_frequency = -1;

    long trajectory_save_frequency = -1;
    std::vector<long> group_save_frequencies = {};

    // One of lin or log
    std::string energy_save_style = "lin";
    std::string trajectory_save_style = "lin";
    std::vector<std::string> group_save_styles = {};

    bool overwrite = true;
    long precision = 3;
    std::string trajectories_dir_name = "trajectories";
    std::string system_dir_name = "system";
    std::string restart_dir_name = "restart";
    std::string init_dir_name = "init";
    std::string group_dir_name = "group";
    std::string indexed_file_prefix = "t";
    std::string delimeter = ",";
    std::string file_extension = ".csv";
};


class FileManager {
protected:
    // maybe it should be something like:
    // list of orchestrators
    // each one has a set of log names
    // "save group"
    // energy, states, extra (similar to trajectories)
    // can have a save by group id method
    // each group has a save frequency

    // save configuration (restart)

    // structure:
    // system (energy, most recent config, initial config, parameters, script details + arguments)
    // trajectories (config data)
    // group 0 (misc config data)
    // ....
    // group N (misc config data)

    // each group has own saving frequency and saving scheme: log or lin

    // log saving: find number of decades spanned by run-length, get equal number of values in each decade

    // maybe should rethink the orchestrator so that at a given save step - all the RELEVANT lognames are passed, reduced to unique set, and orchestrated
    // maybe this means that the orchestrator should be constructed when the save step is active

    // default behaviour should be to open a directory and overwrite everything inside of it
    // overwriting should be turned off if continuing a run
    // this should work with the open energy file functionality and should determine if energy_file_has_header

    // each particle should have a fundamental and derived values - if fundamental is missing, throw error, if derived is missing, derive
    // list of strings: fundamental = [d_vertex_pos, d_vertex_vel], derived = [d_pos, d_vel]

    // define the information needed for the restart (particle.fundamental_values)
    // probably also need to pass an integrator object to the save config

    Orchestrator orchestrator;

    FileManagerConfig config;

    std::ifstream input_file;  // multi-purpose (configs, etc.)
    std::ofstream output_file;  // multi-purpose (configs, etc.)
    std::ofstream energy_file;  // only for energy

    bool energy_file_has_header = false;

public:
    FileManager(Particle& particle, FileManagerConfig& config);
    ~FileManager();

    // TODO: tabular data should be csv
    // TODO: file format should also be a configuration option
    // TOOD: write docstrings


    std::string trajectory_path;
    std::string system_path;
    std::string restart_path;
    std::string init_path;
    std::string group_path;

    // TODO: can the table writing be generalized to include console logger behavior?

    void initialize_step_managers();


    void orchestrate(long step);


    void write_energy_header();

    void write_energy_values(long step);

    // TODO: read / write matrix (nxm) (generalizes scalar and vector storage)
    // just specify size and type
    thrust::host_vector<double> 

    // TODO: read / write parameters (more generally, struct) (construct particle objects from the output of this - should return a particle constructor struct)
    // TODO: write energy
};

#endif /* FILE_MANAGER_H */
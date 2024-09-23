#ifndef ENERGY_LOG_H
#define ENERGY_LOG_H

#include "base_log_groups.h"
#include "utils.h"
#include <fstream>
#include <iostream>


class EnergyLog : public MacroLog {
private:
    std::string delimiter = ",";
    std::ofstream log_file;
    std::string file_name;
    long precision = 16;

public:
    EnergyLog(LogGroupConfig log_group_config, Orchestrator& orchestrator, const std::string& file_name);
    ~EnergyLog();

    bool has_header = false;
    void write_header();
    void log(long step) final;
    static EnergyLog from_names_lin(Orchestrator& orchestrator, const std::string& file_name, std::vector<std::string> log_names, long num_steps, long num_saves);
    static EnergyLog from_names_log(Orchestrator& orchestrator, const std::string& file_name, std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade);
};


#endif /* ENERGY_LOG_H */

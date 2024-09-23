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
};

#endif /* ENERGY_LOG_H */

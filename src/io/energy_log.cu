#include "../../include/io/base_log_groups.h"
#include "../../include/io/energy_log.h"
#include "../../include/io/utils.h"

EnergyLog::EnergyLog(LogGroupConfig log_group_config, Orchestrator& orchestrator, const std::string& file_name, bool overwrite)
    : MacroLog(log_group_config, orchestrator) {
    this->file_name = file_name;
    log_file = open_output_file(file_name, overwrite);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: EnergyLog: could not open file: " << file_name << std::endl;
        exit(1);
    }
    if (log_file.tellp() != 0) {
        has_header = true;
    } else {
        write_header();
    }
}

EnergyLog::~EnergyLog() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

void EnergyLog::write_header() {
    for (size_t i = 0; i < config.log_names.size(); ++i) {
        log_file << config.log_names[i];
        if (i < config.log_names.size() - 1) {
            log_file << delimiter;
        }
    }
    log_file << "\n";
    log_file.flush();
}

void EnergyLog::log(long step) {
    for (size_t i = 0; i < config.log_names.size(); ++i) {
        double value = orchestrator.get_value<double>(unmodified_log_names[i], step);
        if (log_name_is_modified(config.log_names[i])) {
            std::string modifier = get_modifier(config.log_names[i]);
            value = orchestrator.apply_modifier(modifier, value);
        }
        log_file << std::fixed << std::setprecision(precision) << value;
        if (i < config.log_names.size() - 1) {
            log_file << delimiter;
        }
    }
    log_file << "\n";
    log_file.flush();
}

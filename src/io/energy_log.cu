#include "../../include/io/base_log_groups.h"
#include "../../include/io/energy_log.h"
#include "../../include/io/utils.h"

EnergyLog::EnergyLog(LogGroupConfig log_group_config, Orchestrator& orchestrator, const std::string& file_name)
    : MacroLog(log_group_config, orchestrator) {
    this->file_name = file_name;
    log_file.open(file_name);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: EnergyLog: could not open file: " << file_name << std::endl;
        exit(1);
    }
    if (log_file.tellp() != 0) {
        has_header = true;
    }
    else {
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


EnergyLog EnergyLog::from_names_lin(Orchestrator& orchestrator, const std::string& file_name, std::vector<std::string> log_names, long num_steps, long num_saves) {
    LogGroupConfig config = config_from_names_lin(log_names, num_steps, num_saves);
    return EnergyLog(config, orchestrator, file_name);
}

EnergyLog EnergyLog::from_names_log(Orchestrator& orchestrator, const std::string& file_name, std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade) {
    LogGroupConfig config = config_from_names_log(log_names, num_steps, num_saves, min_save_decade);
    return EnergyLog(config, orchestrator, file_name);
}

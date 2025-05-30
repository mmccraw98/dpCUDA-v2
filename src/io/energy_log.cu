#include "../../include/io/base_log_groups.h"
#include "../../include/io/energy_log.h"
#include "../../include/io/io_utils.h"

EnergyLog::EnergyLog(ConfigDict log_group_config, Orchestrator& orchestrator, const std::string& file_name, bool overwrite)
    : ScalarLog(log_group_config, orchestrator) {
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
    for (size_t i = 0; i < log_names.size(); ++i) {
        log_file << log_names[i];
        if (i < log_names.size() - 1) {
            log_file << delimiter;
        }
    }
    log_file << "\n";
    log_file.flush();
}

void EnergyLog::log(long step) {  // TODO: operate on gathered data
    for (size_t i = 0; i < log_names.size(); ++i) {
        double value = gathered_data[log_names[i]];
        log_file << std::scientific << std::setprecision(precision) << value;
        if (i < log_names.size() - 1) {
            log_file << delimiter;
        }
    }
    log_file << "\n";
    log_file.flush();
    gathered_data.clear();
}

std::unique_ptr<BaseLogGroup> EnergyLog::snapshot() const {
    // Create a new EnergyLog instance using a custom constructor;
    // set "overwrite" to false so that the file is opened in append mode.
    auto snap = std::unique_ptr<EnergyLog>(
        new EnergyLog(this->config, this->orchestrator, this->file_name, false)
    );
    // Copy the gathered data and any other safe-to-copy members.
    snap->gathered_data = this->gathered_data;
    snap->has_header = this->has_header;
    return snap;
}
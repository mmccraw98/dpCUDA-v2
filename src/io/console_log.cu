#include "../../include/io/console_log.h"

ConsoleLog::ConsoleLog(LogGroupConfig config, Orchestrator& orchestrator)
    : ScalarLog(config, orchestrator) {
}

ConsoleLog::~ConsoleLog() {
}

void ConsoleLog::write_header() {
    std::ostringstream out;
    out << std::string(this->width * config.log_names.size() + (config.log_names.size() - 1), '_') << std::endl;
    for (int i = 0; i < config.log_names.size(); i++) {
        out << std::setw(this->width) << config.log_names[i];
        if (i < config.log_names.size() - 1) {
            out << this->delimiter;
        }
    }
    out << std::endl << std::string(this->width * config.log_names.size() + (config.log_names.size() - 1), '_') << std::endl;
    std::cout << out.str();
}

void ConsoleLog::log(long step) {  // TODO: operate on gathered data
    if (last_header_log > header_log_freq) {
        write_header();
        last_header_log = 0;
    }
    last_header_log += 1;
    std::ostringstream out;
    for (int i = 0; i < config.log_names.size(); i++) {
        double value = gathered_data[config.log_names[i]];
        out << std::setw(width) << std::scientific << std::setprecision(precision) << value;
        if (i < config.log_names.size() - 1) {
            out << delimiter;
        }
    }
    std::cout << out.str() << std::endl;
}

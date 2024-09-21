#include "../../include/io/utils.h"
#include "../../include/io/log_manager.h"
    
LogGroup::LogGroup(std::vector<std::string> log_names, LogManagerConfig config) {
    this->log_names = log_names;
    this->log_manager = LogManager(config);
}

LogGroup::~LogGroup() {
}   

void LogGroup::check_log_status(long step) {
    this->should_log = this->log_manager.should_log(step);
}

std::vector<std::string> LogGroup::get_log_names() {
    return this->log_names;
}

// initialize the step manager and add the log names
LogGroup from_num_saves_lin(long num_saves, long max_steps, std::vector<std::string> log_names) {
    LogManagerConfig config;
    config.save_style = "lin";
    config.save_freq = static_cast<long>(static_cast<double>(max_steps) / static_cast<double>(num_saves));
    return LogGroup(log_names, config);
}

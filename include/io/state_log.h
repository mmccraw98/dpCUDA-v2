#ifndef STATE_LOG_H
#define STATE_LOG_H

#include "base_log_groups.h"
#include <iostream>

class StateLog : public BaseLogGroup {
private:
    int precision = 3;

public:
    StateLog(LogGroupConfig config, Orchestrator& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix);
    ~StateLog();

    void write_header();
    void log(long step) final;
    static StateLog from_names_lin(Orchestrator& orchestrator, std::vector<std::string> log_names, long num_steps, long num_saves, const std::string& root_path, const std::string& indexed_file_prefix);
    static StateLog from_names_log(Orchestrator& orchestrator, std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, const std::string& root_path, const std::string& indexed_file_prefix);
};

#endif /* STATE_LOG_H */
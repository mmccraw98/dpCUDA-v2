#ifndef STATE_LOG_H
#define STATE_LOG_H

#include "base_log_groups.h"
#include "utils.h"
#include <iostream>

class StateLog : public BaseLogGroup {
private:
    int precision = 3;
    std::string root_path;
    std::string indexed_file_prefix;
    std::string extension;

public:
    StateLog(LogGroupConfig config, Orchestrator& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix, const std::string& extension);
    ~StateLog();

    void write_header();
    void log(long step) final;
};

#endif /* STATE_LOG_H */
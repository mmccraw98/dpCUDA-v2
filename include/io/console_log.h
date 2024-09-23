#ifndef CONSOLE_LOG_H
#define CONSOLE_LOG_H

#include "base_log_groups.h"
#include <iostream>

class ConsoleLog : public MacroLog {
private:
    std::string delimeter = "|";
    int precision = 3;
    int width = 10;
    int header_log_freq = 10;
    int last_header_log = header_log_freq + 1;

public:
    ConsoleLog(LogGroupConfig config, Orchestrator& orchestrator);
    ~ConsoleLog();

    void write_header();
    void log(long step) final;
};

#endif /* CONSOLE_LOG_H */
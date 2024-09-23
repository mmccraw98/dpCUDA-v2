#ifndef BASE_LOG_GROUPS_H
#define BASE_LOG_GROUPS_H

#include <string>
#include <vector>
#include "orchestrator.h"

struct LogGroupConfig {
    std::vector<std::string> log_names;
    std::string save_style;  // "lin" or "log"
    long save_freq = 1;  // save frequency (does nothing for log)
    long reset_save_decade;  // the maximum decade before the save frequency is reset
    long min_save_decade;  // the minimum save frequency
    long multiple = 0;  // the current multiple of the reset_save_decade
    long decade = 10;  // the decade to multiply the save frequency by

    ~LogGroupConfig() {
        log_names.clear();
    }
};

LogGroupConfig config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves);
LogGroupConfig config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade);

class BaseLogGroup {
protected:
    LogGroupConfig config;
    Orchestrator& orchestrator;
    std::vector<std::string> unmodified_log_names;

public:
    BaseLogGroup(LogGroupConfig config, Orchestrator& orchestrator);
    virtual ~BaseLogGroup();

    void update_log_status(long step);
    bool should_log = false;
    virtual void log(long step) = 0;
};


class MacroLog : public BaseLogGroup {
protected:
    std::vector<std::string> unmodified_log_names;
    std::string delimiter;
    std::string modifier = "/";
    long precision;
    long width;

public:
    MacroLog(LogGroupConfig config, Orchestrator& orchestrator);
    virtual ~MacroLog();

    bool log_name_is_modified(std::string log_name);
    std::vector<std::string> get_unmodified_log_names();
    virtual void log(long step) = 0;
    std::string get_modifier(std::string log_name);
};


#endif /* BASE_LOG_GROUPS_H */
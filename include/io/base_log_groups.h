#ifndef BASE_LOG_GROUPS_H
#define BASE_LOG_GROUPS_H

#include <string>
#include <vector>
#include "orchestrator.h"

#include <nlohmann/json.hpp>

struct LogGroupConfig {
    std::vector<std::string> log_names;
    std::string save_style;  // "lin" or "log"
    long save_freq = 1;  // save frequency (does nothing for log)
    long reset_save_decade = 10;  // the maximum decade before the save frequency is reset
    long min_save_decade = 10;  // the minimum save frequency
    long multiple = 0;  // the current multiple of the reset_save_decade
    long decade = 10;  // the decade to multiply the save frequency by
    std::string group_name;

    ~LogGroupConfig() {
        log_names.clear();
    }

    nlohmann::json to_json() {
        return nlohmann::json{
            {"log_names", log_names},
            {"save_style", save_style},
            {"save_freq", save_freq},
            {"reset_save_decade", reset_save_decade},
            {"min_save_decade", min_save_decade},
            {"group_name", group_name}
        };
    }

    static LogGroupConfig from_json(const nlohmann::json& j) {
        LogGroupConfig config;
        config.log_names = j.at("log_names").get<std::vector<std::string>>();
        config.save_style = j.at("save_style").get<std::string>();
        config.save_freq = j.at("save_freq").get<long>();
        config.reset_save_decade = j.at("reset_save_decade").get<long>();
        config.min_save_decade = j.at("min_save_decade").get<long>();
        config.group_name = j.at("group_name").get<std::string>();
        return config;
    }
};


LogGroupConfig config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name);
LogGroupConfig config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name);
LogGroupConfig config_from_names_lin_everyN(std::vector<std::string> log_names, long save_freq, std::string group_name);

class BaseLogGroup {
protected:
    Orchestrator& orchestrator;
    std::vector<std::string> unmodified_log_names;

public:
    BaseLogGroup(LogGroupConfig config, Orchestrator& orchestrator);
    virtual ~BaseLogGroup();

    LogGroupConfig config;
    
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
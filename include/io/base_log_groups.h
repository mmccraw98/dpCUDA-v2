#ifndef BASE_LOG_GROUPS_H
#define BASE_LOG_GROUPS_H

#include <string>
#include <vector>
#include "orchestrator.h"

#include <nlohmann/json.hpp>

/**
 * @brief The configuration for a log group.
 * 
 * This struct contains all the configuration options for a log group.
 * Determines when a log group should be logged, what it should log, and how it should log it.
 */
struct LogGroupConfig {
    std::vector<std::string> log_names;  // the names of the variables to log
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

    /**
     * @brief Convert the LogGroupConfig to a JSON object.
     * 
     * This function converts the LogGroupConfig to a JSON object using the nlohmann::json library.
     * 
     * @return The LogGroupConfig as a JSON object.
     */
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

    /**
     * @brief Create a LogGroupConfig from a JSON object.
     * 
     * This function creates a LogGroupConfig from a JSON object using the nlohmann::json library.
     * 
     * @param j The JSON object to create the LogGroupConfig from.
     * @return The LogGroupConfig created from the JSON object.
     */
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


/**
 * @brief Create a LogGroupConfig for linear saving.
 * 
 * This function creates a LogGroupConfig for linear saving with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param num_steps The total number of steps.
 * @param num_saves The number of saves.
 * @param group_name The name of the log group.
 * @return The LogGroupConfig for linear saving.
 */
LogGroupConfig config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name);

/**
 * @brief Create a LogGroupConfig for logarithmic saving.
 * 
 * This function creates a LogGroupConfig for logarithmic saving with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param num_steps The total number of steps.
 * @param num_saves The number of saves.  Works out to set the maximum decade before the save frequency is reset to the min_save_decade.
 * @param min_save_decade The smallest decade to save.
 * @param group_name The name of the log group.
 * @return The LogGroupConfig for logarithmic saving.
 */
LogGroupConfig config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name);

/**
 * @brief Create a LogGroupConfig for linear saving every N steps.
 * 
 * This function creates a LogGroupConfig for linear saving every N steps with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param save_freq The frequency at which to save.
 * @param group_name The name of the log group.
 * @return The LogGroupConfig for linear saving every N steps.
 */
LogGroupConfig config_from_names_lin_everyN(std::vector<std::string> log_names, long save_freq, std::string group_name);


/**
 * @brief Create a LogGroupConfig for logging the particle state (particle radii, etc.).  Done only once.
 * 
 * @param log_names The names of the variables to log.
 * @param group_name The name of the log group.
 * @return The LogGroupConfig for logging the particle state.
 */
LogGroupConfig config_from_names(std::vector<std::string> log_names, std::string group_name);

/**
 * @brief Base class for all log groups.
 * 
 * This class is the base class for all log groups.
 * It contains the common functionality for all log groups.
 */
class BaseLogGroup {
protected:
    Orchestrator& orchestrator;  // the orchestrator that manages the interface between the logs and the simulation objects
    std::vector<std::string> unmodified_log_names;  // the unmodified log names

public:
    BaseLogGroup(LogGroupConfig config, Orchestrator& orchestrator);
    virtual ~BaseLogGroup();

    LogGroupConfig config;  // the configuration for the log group
    
    bool should_log = false;  // whether the log group should log

    /**
     * @brief Update the log status.
     * 
     * This function updates the log status based on the current step.
     * 
     * @param step The current step.
     */
    void update_log_status(long step);

    /**
     * @brief Log the current state.
     * 
     * This function logs the current state.
     * 
     * @param step The current step.
     */
    virtual void log(long step) = 0;
};


/**
 * @brief MacroLog class.
 * 
 * This class is the base class for all macro log groups.
 * It contains the common functionality for all macro log groups (console, energy, etc.)
 */
class MacroLog : public BaseLogGroup {
protected:
    std::vector<std::string> unmodified_log_names;  // the list of log names without modifiers
    std::string delimiter;  // the delimiter to use when logging
    std::string modifier = "/";  // append to log names to divide them by values
    long precision;  // the precision to use when logging
    long width;  // the width to use when logging

public:
    MacroLog(LogGroupConfig config, Orchestrator& orchestrator);
    virtual ~MacroLog();

    bool log_name_is_modified(std::string log_name);  // check if the log name is modified (contains a modifier)
    std::vector<std::string> get_unmodified_log_names();  // get the unmodified log names
    virtual void log(long step) = 0;  // log the current state
    std::string get_modifier(std::string log_name);  // get the modifier from the log name
};


#endif /* BASE_LOG_GROUPS_H */
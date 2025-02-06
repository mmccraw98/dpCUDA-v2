#pragma once

#include <string>
#include <vector>
#include "orchestrator.h"
#include <set>

#include <memory>

#include <nlohmann/json.hpp>

#include "../utils/config_dict.h"

inline ConfigDict get_log_group_config_dict(){
    ConfigDict config;
    config["log_names"] = std::vector<std::string>();
    config["save_style"] = "lin";
    config["save_freq"] = 1;
    config["reset_save_decade"] = 10;
    config["min_save_decade"] = 10;
    config["multiple"] = 0;
    config["group_name"] = "none";
    config["decade"] = 10;
    return config;
}

/**
 * @brief Create a LogGroupConfig for linear saving.
 * 
 * This function creates a LogGroupConfig for linear saving with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param num_steps The total number of steps.
 * @param num_saves The number of saves.
 * @param group_name The name of the log group.
 * @return The LogGroupConfigDict for linear saving.
 */
ConfigDict config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name);

/**
 * @brief Create a LogGroupConfigDict for logarithmic saving.
 * 
 * This function creates a LogGroupConfigDict for logarithmic saving with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param num_steps The total number of steps.
 * @param num_saves The number of saves.  Works out to set the maximum decade before the save frequency is reset to the min_save_decade.
 * @param min_save_decade The smallest decade to save.
 * @param group_name The name of the log group.
 * @return The LogGroupConfigDict for logarithmic saving.
 */
ConfigDict config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name);

/**
 * @brief Create a LogGroupConfigDict for linear saving every N steps.
 * 
 * This function creates a LogGroupConfigDict for linear saving every N steps with the given parameters.
 * 
 * @param log_names The names of the variables to log.
 * @param save_freq The frequency at which to save.
 * @param group_name The name of the log group.
 * @return The LogGroupConfigDict for linear saving every N steps.
 */
ConfigDict config_from_names_lin_everyN(std::vector<std::string> log_names, long save_freq, std::string group_name);


/**
 * @brief Create a LogGroupConfigDict for logging the particle state (particle radii, etc.).  Done only once.
 * 
 * @param log_names The names of the variables to log.
 * @param group_name The name of the log group.
 * @return The LogGroupConfigDict for logging the particle state.
 */
ConfigDict config_from_names(std::vector<std::string> log_names, std::string group_name);

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
    std::set<std::string> dependencies;

public:
    BaseLogGroup(ConfigDict config, Orchestrator& orchestrator);
    virtual ~BaseLogGroup();

    ConfigDict config;  // the configuration for the log group

    std::vector<std::string> log_names;
    std::string save_style;
    long save_freq;
    long reset_save_decade;
    long min_save_decade;
    long multiple;
    std::string group_name;
    long decade;
    
    bool should_log = false;  // whether the log group should log

    virtual void define_dependencies();

    void handle_dependencies();

    bool has_dependencies = false;

    bool parallel = false;

    /**
     * @brief Update the log status.
     * 
     * This function updates the log status based on the current step.
     * 
     * @param step The current step.
     */
    void update_log_status(long step);

    virtual void gather_data(long step) = 0;  // gather the data for the log group

    /**
     * @brief Log the current state.
     * 
     * This function logs the current state.
     * 
     * @param step The current step.
     */
    virtual void log(long step) = 0;

    /**
     * @brief Create a snapshot (deep copy) of the logging data.
     * 
     * This method should return a unique_ptr to a copy of the log group containing
     * the data to be logged so that asynchronous tasks do not operate on data that
     * may later be overwritten.
     *
     * @return A unique_ptr to the snapshot.
     */
    virtual std::unique_ptr<BaseLogGroup> snapshot() const = 0;
};


/**
 * @brief ScalarLog class.
 * 
 * This class is the base class for all macro log groups.
 * It contains the common functionality for all macro log groups (console, energy, etc.)
 */
class ScalarLog : public BaseLogGroup {
protected:
    std::vector<std::string> unmodified_log_names;  // the list of log names without modifiers
    std::unordered_map<std::string, double> gathered_data;
    std::string delimiter;  // the delimiter to use when logging
    std::string modifier = "/";  // append to log names to divide them by values
    long precision;  // the precision to use when logging
    long width;  // the width to use when logging

public:
    ScalarLog(ConfigDict config, Orchestrator& orchestrator);
    virtual ~ScalarLog();

    bool is_modified(std::string log_name);  // check if the log name is modified (contains a modifier)
    std::vector<std::string> get_unmodified_log_names();  // get the unmodified log names
    void define_dependencies() override;
    void gather_data(long step) override;
    virtual void log(long step) = 0;  // log the current state
    std::string get_modifier(std::string log_name);  // get the modifier from the log name
};


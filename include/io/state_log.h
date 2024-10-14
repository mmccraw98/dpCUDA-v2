#ifndef STATE_LOG_H
#define STATE_LOG_H

#include "base_log_groups.h"
#include "utils.h"
#include "../include/constants.h"
#include <iostream>

/**
 * @brief StateLog class.
 * 
 * This class is used to log the state of the system.
 */
class StateLog : public BaseLogGroup {
private:
    int precision = DECIMAL_PRECISION;  // the precision to use when logging
    std::string root_path;  // the root path to log to
    std::string indexed_file_prefix;  // the indexed file prefix to use when logging
    std::string extension;  // the extension to use when logging

public:
    StateLog(LogGroupConfig config, Orchestrator& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix, const std::string& extension);
    ~StateLog();

    /**
     * @brief Write the header to the log file.
     */
    void write_header();

    /**
     * @brief Write the all the vectors to their files in the root_path.
     * 
     * @param root_path The root path to write to.
     */
    void write_values(std::filesystem::path root_path);

    /**
     * @brief Log the current state of the system by writing all the vectors to their files in a subdirectory prefixed with the step.
     * 
     * @param step The current step.
     */
    void log(long step) final;

    /**
     * @brief Write the state of the system.  Used for saving initial conditions.
     */
    void write_state();
};

#endif /* STATE_LOG_H */
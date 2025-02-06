#pragma once

#include "base_log_groups.h"
#include <iostream>

/**
 * @brief ConsoleLog is a class that handles logging to the console.
 * 
 */
class ConsoleLog : public ScalarLog {
private:
    std::string delimiter = "|";  // delimiter for the console log
    int precision = 3;  // precision for the console log
    int width = 10;  // width of each column
    int header_log_freq = 10;  // frequency for writing the header
    int last_header_log = header_log_freq + 1;  // last time the header was logged

public:
    /**
     * @brief Construct a new Console Log object
     * 
     * @param config The configuration for the log group
     * @param orchestrator The orchestrator that handles the pre-requisite calculations
     */
    ConsoleLog(ConfigDict config, Orchestrator& orchestrator);
    ~ConsoleLog();

    /**
     * @brief Writes the header to the console log
     * 
     */
    void write_header();

    /**
     * @brief Logs the values to the console log
     * 
     * @param step The current step
     */
    void log(long step) final;

    /**
     * @brief Create a snapshot of this ConsoleLog.
     * @return A unique_ptr containing a copy of this ConsoleLog.
     */
    virtual std::unique_ptr<BaseLogGroup> snapshot() const override;
};

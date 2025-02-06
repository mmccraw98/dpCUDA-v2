#pragma once

#include "base_log_groups.h"
#include "io_utils.h"
#include "../include/constants.h"
#include <fstream>
#include <iostream>

/**
 * @brief EnergyLog is a class that writes the energy values to delimited file.
 * 
 * @param log_group_config The configuration for the log group
 * @param orchestrator The orchestrator that handles the pre-requisite calculations
 * @param file_name The name of the file to write to
 * @param overwrite Whether to overwrite the file if it exists
 */
class EnergyLog : public ScalarLog {
private:
    std::string delimiter = ",";  // delimiter for the csv file
    std::ofstream log_file;  // the file stream
    std::string file_name;  // the name of the file
    long precision = DECIMAL_PRECISION;  // the precision of the numbers in the file

public:
    /**
     * @brief Construct a new Energy Log object
     * 
     * @param log_group_config The configuration for the log group
     * @param orchestrator The orchestrator that handles the pre-requisite calculations
     * @param file_name The name of the file to write to
     * @param overwrite Whether to overwrite the file if it exists
     */
    EnergyLog(ConfigDict log_group_config, Orchestrator& orchestrator, const std::string& file_name, bool overwrite);
    ~EnergyLog();

    bool has_header = false;  // checks whether the file has a header in its first row
    void write_header();  // writes the header to the file
    void log(long step) final;  // logs the values to the file

    /**
     * @brief Create a snapshot of this EnergyLog.
     * @return A unique_ptr containing a copy of this EnergyLog.
     */
    virtual std::unique_ptr<BaseLogGroup> snapshot() const override;
};


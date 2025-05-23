#pragma once

#include "base_log_groups.h"
#include "io_utils.h"
#include "../include/constants.h"
#include "../include/data/array_data.h"
#include <iostream>

/**
 * @brief StateLog class.
 * 
 * This class is used to log the state of the system.
 */
class StateLog : public BaseLogGroup {
private:
    long precision = DECIMAL_PRECISION;  // the precision to use when logging
    std::string root;  // the root path to log to
    std::string indexed_file_prefix;  // the indexed file prefix to use when logging
    std::string extension;  // the extension to use when logging
    std::unordered_map<std::string, ArrayData> gathered_data;
    std::unordered_map<std::string, ArrayData> reorder_index_data;  // key: index name, value: index arraydata
    bool is_restart;

public:
    StateLog(ConfigDict config, Orchestrator& orchestrator, const std::string& root, const std::string& indexed_file_prefix, const std::string& extension, bool is_restart = false);
    ~StateLog();

    // TODO: add sorting by particle index
    // TODO: add grouping together 2d data

    void gather_data(long step) override;

    /**
     * @brief Write the header to the log file.
     */
    void write_header();

    /**
     * @brief Write the all the vectors to their files in the root.
     * 
     * @param root The root path to write to.
     */
    void write_values(const std::filesystem::path& root, long step);

    /**
     * @brief Log the current state of the system by writing all the vectors to their files in a subdirectory prefixed with the step.
     * 
     * @param step The current step.
     */
    void log(long step) final;

    /**
     * @brief Write the state of the system.  Used for saving initial conditions.
     */
    void write_state_to_path(const std::filesystem::path& path);

    /**
     * @brief Create a snapshot of this StateLog.
     * @return A unique_ptr containing a copy of this StateLog.
     */
    virtual std::unique_ptr<BaseLogGroup> snapshot() const override;
};

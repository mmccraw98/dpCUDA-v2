#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include "orchestrator.h"
#include "../particle/particle.h"

struct LoggerConfig {
    long last_header_log_step = 0;
    long header_log_step_frequency = 10;
    long precision = 3;
    long width = 12;
};

class Logger {
protected:
    Orchestrator orchestrator;
    LoggerConfig config;
public:
    Logger(Particle& particle, const std::vector<std::string>& log_names, LoggerConfig config = LoggerConfig());
    ~Logger();

    // TODO: tabular data should be csv
    // TODO: file format should also be a configuration option
    // TOOD: write docstrings

    void write_header();

    void write_values(long step);
};

#endif /* LOGGER_H */
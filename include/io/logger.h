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

class Logger {
protected:
    Particle& particle;
    Orchestrator& orchestrator;

public:
    Logger(Particle& particle, Orchestrator& orchestrator);  // logger accepts an orchestrator - this way there can be a global orchestrator for all logging
    ~Logger();

    // TODO: tabular data should be csv
    // TODO: file format should also be a configuration option
    // TOOD: write docstrings

    // TODO: pass a configuration struct to the logger object to construct it
    long last_header_log_step = 0;
    long header_log_step_frequency = 10;
    long precision = 3;
    long width = 12;

    void write_header();

    void write_values(long step);
};

#endif /* LOGGER_H */
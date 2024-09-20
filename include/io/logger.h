#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "../particle/particle.h"

class Logger {
protected:
    Particle& particle;
    std::vector<std::string> log_names;

public:
    Logger(Particle& particle, const std::vector<std::string>& log_names);
    ~Logger();

    std::vector<std::string> modifiers = {"/"};
    long last_header_log_step = 0;

    void write_header(long width = 12);

    // get value (from name)
    double get_value(const std::string& name);

    // apply modifier (from thing following name i.e. divide by N)
    double apply_modifier(std::string& name, double value);

    void write_values(long step, long width = 12);
};

#endif /* LOGGER_H */
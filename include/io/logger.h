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

    void write_header(long width = 12);

    // get value (from name)

    // apply modifier (from thing following name i.e. divide by N)
    
};

#endif /* LOGGER_H */
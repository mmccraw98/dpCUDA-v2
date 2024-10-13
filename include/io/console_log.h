#ifndef CONSOLE_LOG_H
#define CONSOLE_LOG_H

#include "base_log_groups.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
#include "../../include/integrator/integrator.h"
#include "../../include/integrator/nve.h"
#include <iostream>

template <typename ParticleType, typename IntegratorType>
class ConsoleLog : public MacroLog<ParticleType, IntegratorType> {
private:
    std::string delimiter = "|";
    int precision = 3;
    int width = 10;
    int header_log_freq = 10;
    int last_header_log = header_log_freq + 1;

public:
    ConsoleLog(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator);
    ~ConsoleLog();

    void write_header();
    void log(long step) final;
};

#include "console_log_impl.h"

#endif /* CONSOLE_LOG_H */
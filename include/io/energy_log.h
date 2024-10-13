#ifndef ENERGY_LOG_H
#define ENERGY_LOG_H

#include "base_log_groups.h"
#include "utils.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
#include "../../include/integrator/integrator.h"
#include "../../include/integrator/nve.h"
#include <fstream>
#include <iostream>


template <typename ParticleType, typename IntegratorType>
class EnergyLog : public MacroLog<ParticleType, IntegratorType> {
private:
    std::string delimiter = ",";
    std::ofstream log_file;
    std::string file_name;
    long precision = 16;

public:
    EnergyLog(LogGroupConfig log_group_config, Orchestrator<ParticleType, IntegratorType>& orchestrator, const std::string& file_name, bool overwrite);
    ~EnergyLog();

    bool has_header = false;
    void write_header();
    void log(long step) final;
};

#include "energy_log_impl.h"

#endif /* ENERGY_LOG_H */

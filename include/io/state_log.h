#ifndef STATE_LOG_H
#define STATE_LOG_H

#include "base_log_groups.h"
#include "utils.h"
#include <iostream>

template <typename ParticleType, typename IntegratorType>
class StateLog : public BaseLogGroup<ParticleType, IntegratorType> {
private:
    int precision = 3;
    std::string root_path;
    std::string indexed_file_prefix;
    std::string extension;

public:
    StateLog(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix, const std::string& extension);
    ~StateLog();

    void write_header();
    void log(long step) final;
};


#include "state_log_impl.h"
#endif /* STATE_LOG_H */
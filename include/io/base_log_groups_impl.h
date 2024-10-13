#ifndef BASE_LOG_GROUPS_IMPL_H
#define BASE_LOG_GROUPS_IMPL_H


#include "base_log_groups.h"

LogGroupConfig config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.save_style = "lin";
    config.save_freq = static_cast<long>(num_steps / num_saves);
    config.group_name = group_name;
    return config;
}

LogGroupConfig config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.save_style = "log";
    config.reset_save_decade = static_cast<long>(num_steps / num_saves);
    config.min_save_decade = min_save_decade;
    config.group_name = group_name;
    return config;
}

LogGroupConfig config_from_names_lin_everyN(std::vector<std::string> log_names, long save_freq, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.save_style = "lin";
    config.save_freq = save_freq;
    config.group_name = group_name;
    return config;
}

template <typename ParticleType, typename IntegratorType>
BaseLogGroup<ParticleType, IntegratorType>::BaseLogGroup(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator) : config(config), orchestrator(orchestrator) {
}

template <typename ParticleType, typename IntegratorType>
BaseLogGroup<ParticleType, IntegratorType>::~BaseLogGroup() {
}

template <typename ParticleType, typename IntegratorType>
void BaseLogGroup<ParticleType, IntegratorType>::update_log_status(long step) {
    if (this->config.save_style == "lin") {
        this->should_log = step % this->config.save_freq == 0;
    } else if (this->config.save_style == "log") {
        if (step > this->config.multiple * this->config.reset_save_decade) {
            this->config.save_freq = this->config.min_save_decade;
            this->config.multiple += 1;
        }
        if ((step - (this->config.multiple - 1) * this->config.reset_save_decade) > this->config.save_freq * this->config.decade) {
            this->config.save_freq *= this->config.decade;
        }
        if ((step - (this->config.multiple - 1) * this->config.reset_save_decade) % this->config.save_freq == 0) {
            this->should_log = true;
        } else {
            this->should_log = false;
        }
    } else {
        std::cout << "ERROR: BaseLogGroup::update_log_status: Invalid save style: " << this->config.save_style << std::endl;
        exit(1);
    }
}

template <typename ParticleType, typename IntegratorType>
MacroLog<ParticleType, IntegratorType>::MacroLog(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator)
    : BaseLogGroup<ParticleType, IntegratorType>(config, orchestrator) {
    this->unmodified_log_names = this->get_unmodified_log_names();
}


template <typename ParticleType, typename IntegratorType>
MacroLog<ParticleType, IntegratorType>::~MacroLog() {
}

template <typename ParticleType, typename IntegratorType>
bool MacroLog<ParticleType, IntegratorType>::log_name_is_modified(std::string log_name) {
    return log_name.find(modifier) != std::string::npos;
}

template <typename ParticleType, typename IntegratorType>
std::vector<std::string> MacroLog<ParticleType, IntegratorType>::get_unmodified_log_names() {
    std::vector<std::string> unmodified_log_names;
    for (auto& log_name : this->config.log_names) {
        size_t pos = log_name.find(this->modifier);
        if (pos != std::string::npos) {
            unmodified_log_names.push_back(log_name.substr(0, pos));
        } else {
            unmodified_log_names.push_back(log_name);
        }
    }
    return unmodified_log_names;
}


template <typename ParticleType, typename IntegratorType>
std::string MacroLog<ParticleType, IntegratorType>::get_modifier(std::string log_name) {
    size_t pos = log_name.find(this->modifier);
    if (pos != std::string::npos) {
        return log_name.substr(pos + 1);
    } else {
        return "";
    }
}

#endif // BASE_LOG_GROUPS_IMPL_H
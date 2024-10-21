#include "../../include/io/base_log_groups.h"
#include "../../include/io/orchestrator.h"
#include <string>
#include <iostream>


LogGroupConfig config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.save_style = "lin";
    config.save_freq = static_cast<long>(num_steps / num_saves);
    if (config.save_freq == 0) {
        config.save_freq = 1;
    }
    config.group_name = group_name;
    return config;
}

LogGroupConfig config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.save_style = "log";
    config.reset_save_decade = static_cast<long>(num_steps / num_saves);
    if (config.reset_save_decade == 0) {
        config.reset_save_decade = 1;
    }
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

LogGroupConfig config_from_names(std::vector<std::string> log_names, std::string group_name) {
    LogGroupConfig config;
    config.log_names = log_names;
    config.group_name = group_name;
    return config;
}

BaseLogGroup::BaseLogGroup(LogGroupConfig config, Orchestrator& orchestrator) : config(config), orchestrator(orchestrator) {
}

BaseLogGroup::~BaseLogGroup() {
}

void BaseLogGroup::define_dependencies() {
    for (const std::string& log_name : config.log_names) {
        if (orchestrator.is_dependent(log_name)) {
            dependencies.insert(log_name);
            has_dependencies = true;
        }
    }
}

void BaseLogGroup::handle_dependencies() {
    for (const std::string& log_name : dependencies) {
        orchestrator.handle_dependencies(log_name);
    }
}

void BaseLogGroup::update_log_status(long step) {
    if (config.save_style == "lin") {
        should_log = step % config.save_freq == 0;
    } else if (config.save_style == "log") {
        if (step > config.multiple * config.reset_save_decade) {
            config.save_freq = config.min_save_decade;
            config.multiple += 1;
        }
        if ((step - (config.multiple - 1) * config.reset_save_decade) > config.save_freq * config.decade) {
            config.save_freq *= config.decade;
        }
        if ((step - (config.multiple - 1) * config.reset_save_decade) % config.save_freq == 0) {
            should_log = true;
        } else {
            should_log = false;
        }
    } else {
        std::cout << "ERROR: LogManager::update_log_status: Invalid save style: " << config.save_style << std::endl;
        exit(1);
    }
}

ScalarLog::ScalarLog(LogGroupConfig config, Orchestrator& orchestrator) : BaseLogGroup(config, orchestrator) {
    unmodified_log_names = get_unmodified_log_names();
}

ScalarLog::~ScalarLog() {
}

void ScalarLog::define_dependencies() {
    for (const std::string& log_name : unmodified_log_names) {
        if (orchestrator.is_dependent(log_name)) {
            dependencies.insert(log_name);
            has_dependencies = true;
        }
    }
}

bool ScalarLog::is_modified(std::string log_name) {
    return log_name.find(modifier) != std::string::npos;
}

void ScalarLog::gather_data(long step) {
    for (size_t i = 0; i < unmodified_log_names.size(); ++i) {
        const auto& name = unmodified_log_names[i];
        double value = orchestrator.get_value<double>(name, step);
        if (is_modified(config.log_names[i])) {
            std::string mod = get_modifier(config.log_names[i]);
            value = orchestrator.apply_modifier(mod, value);
        }
        gathered_data[config.log_names[i]] = value;
    }
}

std::vector<std::string> ScalarLog::get_unmodified_log_names() {
    std::vector<std::string> unmodified_log_names;
    for (auto& log_name : config.log_names) {
        size_t pos = log_name.find(modifier);
        if (pos != std::string::npos) {
            unmodified_log_names.push_back(log_name.substr(0, pos));
        } else {
            unmodified_log_names.push_back(log_name);
        }
    }
    return unmodified_log_names;
}

std::string ScalarLog::get_modifier(std::string log_name) {
    size_t pos = log_name.find(modifier);
    if (pos != std::string::npos) {
        return log_name.substr(pos + 1);
    } else {
        return "";
    }
}
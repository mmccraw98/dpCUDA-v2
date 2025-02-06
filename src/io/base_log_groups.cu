#include "../../include/io/base_log_groups.h"
#include "../../include/io/orchestrator.h"
#include <string>
#include <iostream>


ConfigDict config_from_names_lin(std::vector<std::string> log_names, long num_steps, long num_saves, std::string group_name) {
    ConfigDict config = get_log_group_config_dict();
    config["log_names"] = log_names;
    config["save_style"] = "lin";
    config["save_freq"] = static_cast<long>(num_steps / num_saves);
    if (config.at("save_freq").get<long>() == 0) {
        config["save_freq"] = 1;
    }
    config["group_name"] = group_name;
    return config;
}

ConfigDict config_from_names_log(std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade, std::string group_name) {
    ConfigDict config = get_log_group_config_dict();
    config["log_names"] = log_names;
    config["save_style"] = "log";
    config["reset_save_decade"] = static_cast<long>(num_steps / num_saves);
    if (config.at("reset_save_decade").get<long>() == 0) {
        config["reset_save_decade"] = 1;
    }
    config["min_save_decade"] = min_save_decade;
    config["group_name"] = group_name;
    return config;
}

ConfigDict config_from_names_lin_everyN(std::vector<std::string> log_names, long save_freq, std::string group_name) {
    ConfigDict config = get_log_group_config_dict();
    config["log_names"] = log_names;
    config["save_style"] = "lin";
    config["save_freq"] = save_freq;
    config["group_name"] = group_name;
    return config;
}

ConfigDict config_from_names(std::vector<std::string> log_names, std::string group_name) {
    ConfigDict config = get_log_group_config_dict();
    config["log_names"] = log_names;
    config["group_name"] = group_name;
    return config;
}

BaseLogGroup::BaseLogGroup(ConfigDict config, Orchestrator& orchestrator) : config(config), orchestrator(orchestrator) {
    log_names = config.at("log_names").get<std::vector<std::string>>();
    save_style = config.at("save_style").get<std::string>();
    save_freq = config.at("save_freq").get<long>();
    reset_save_decade = config.at("reset_save_decade").get<long>();
    min_save_decade = config.at("min_save_decade").get<long>();
    multiple = config.at("multiple").get<long>();
    decade = config.at("decade").get<long>();
    group_name = config.at("group_name").get<std::string>();
}

BaseLogGroup::~BaseLogGroup() {
}

void BaseLogGroup::define_dependencies() {
    for (const std::string& log_name : config.at("log_names").get<std::vector<std::string>>()) {
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
    if (save_style == "lin") {
        should_log = step % save_freq == 0;
    } else if (save_style == "log") {
        if (step > multiple * reset_save_decade) {
            save_freq = min_save_decade;
            multiple += 1;
        }
        if ((step - (multiple - 1) * reset_save_decade) > save_freq * decade) {
            save_freq *= decade;
        }
        if ((step - (multiple - 1) * reset_save_decade) % save_freq == 0) {
            should_log = true;
        } else {
            should_log = false;
        }
    } else {
        std::cout << "ERROR: LogManager::update_log_status: Invalid save style: " << config.at("save_style").get<std::string>() << std::endl;
        exit(1);
    }
}

ScalarLog::ScalarLog(ConfigDict config, Orchestrator& orchestrator) : BaseLogGroup(config, orchestrator) {
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
        if (is_modified(log_names[i])) {
            std::string mod = get_modifier(log_names[i]);
            value = orchestrator.apply_modifier(mod, value);
        }
        gathered_data[log_names[i]] = value;
    }
}

std::vector<std::string> ScalarLog::get_unmodified_log_names() {
    std::vector<std::string> unmodified_log_names;
    for (auto& log_name : log_names) {
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
#include "../../include/io/base_log_groups.h"
#include "../../include/io/orchestrator.h"
#include <string>
#include <iostream>

BaseLogGroup::BaseLogGroup(LogGroupConfig config, Orchestrator& orchestrator) : config(config), orchestrator(orchestrator) {
}

BaseLogGroup::~BaseLogGroup() {
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





MacroLog::MacroLog(LogGroupConfig config, Orchestrator& orchestrator) : BaseLogGroup(config, orchestrator) {
    unmodified_log_names = get_unmodified_log_names();
}

MacroLog::~MacroLog() {
}

bool MacroLog::log_name_is_modified(std::string log_name) {
    return log_name.find(modifier) != std::string::npos;
}

std::vector<std::string> MacroLog::get_unmodified_log_names() {
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

std::string MacroLog::get_modifier(std::string log_name) {
    size_t pos = log_name.find(modifier);
    if (pos != std::string::npos) {
        return log_name.substr(pos + 1);
    } else {
        return "";
    }
}
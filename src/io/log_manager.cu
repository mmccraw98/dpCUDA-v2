#include "../../include/io/log_manager.h"
#include <string>
#include <vector>
#include <iostream>

LogManager::LogManager(LogManagerConfig config) {
    if (config.save_style != "lin" && config.save_style != "log") {
        std::cerr << "LogManager::LogManager: Invalid save style: " << config.save_style << std::endl;
        exit(EXIT_FAILURE);
    }
    this->config = config;
    this->save_freq_multiple = 1;
}

LogManager::~LogManager() {
}

bool LogManager::should_log(long step) {
    if (config.save_style == "lin") {
        return step % config.save_freq == 0;
    } else if (config.save_style == "log") {
        if (step > save_freq_multiple * config.save_freq) {
            config.save_freq = config.min_save_decade;
            save_freq_multiple += 1;
        }
        if ((step - (save_freq_multiple - 1) * config.save_freq) > config.save_freq * decade) {
            config.save_freq *= decade;
        }
        if ((step - (save_freq_multiple - 1) * config.save_freq) % config.save_freq == 0) {
            return true;
        }
    }
    return false;
}

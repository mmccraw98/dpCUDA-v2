#include "../../include/io/console_log.h"

ConsoleLog::ConsoleLog(LogGroupConfig config, Orchestrator& orchestrator)
    : MacroLog(config, orchestrator) {
}

ConsoleLog::~ConsoleLog() {
}

void ConsoleLog::write_header() {
    std::ostringstream out;
    out << std::string(this->width * config.log_names.size() + (config.log_names.size() - 1), '_') << std::endl;
    for (int i = 0; i < config.log_names.size(); i++) {
        out << std::setw(this->width) << config.log_names[i];
        if (i < config.log_names.size() - 1) {
            out << this->delimiter;
        }
    }
    out << std::endl << std::string(this->width * config.log_names.size() + (config.log_names.size() - 1), '_') << std::endl;
    std::cout << out.str();
}

void ConsoleLog::log(long step) {
    if (last_header_log > header_log_freq) {
        write_header();
        last_header_log = 0;
    }
    last_header_log += 1;
    std::ostringstream out;
    for (int i = 0; i < config.log_names.size(); i++) {
        double value = orchestrator.get_value<double>(unmodified_log_names[i], step);
        if (log_name_is_modified(config.log_names[i])) {
            std::string modifier = get_modifier(config.log_names[i]);
            value = orchestrator.apply_modifier(modifier, value);
        }
        out << std::setw(width) << std::scientific << std::setprecision(precision) << value;
        if (i < config.log_names.size() - 1) {
            out << delimiter;
        }
    }
    std::cout << out.str() << std::endl;
}



// void FileManager::write_energy_values(long step) {
//     energy_orchestrator.precalculate();
//     if (!energy_file_has_header) {write_header();}
//     std::cout << std::setw(width) << step << delimiter << std::setw(width);
//     for (long i = 0; i < orchestrator.log_names.size(); i++) {
//         double value = orchestrator.get_value(orchestrator.log_names[i]);
//         value = orchestrator.apply_modifier(orchestrator.log_names[i], value);
//         std::cout << std::setw(width) << std::scientific << std::setprecision(precision) << value;
//         if (i < orchestrator.log_names.size() - 1) {
//             std::cout << delimiter;
//         }
//     }
//     std::cout << std::endl;
// }


// void Logger::write_header() {
//     long num_names = orchestrator.log_names.size();
//     long total_width = (config.width + 3) * (num_names + 1) - 1;
//     std::cout << std::string(total_width, '_') << std::endl;
//     std::cout << std::setw(config.width) << "step" << " | ";
    
//     for (long i = 0; i < num_names; i++) {
//         std::cout << std::setw(config.width) << orchestrator.log_names[i];
//         if (i < num_names - 1) {
//             std::cout << " | ";
//         }
//     }

//     std::cout << std::endl;
//     std::cout << std::string(total_width, '_') << std::endl;
// }

// void Logger::write_values(long step) {
//     orchestrator.precalculate();
//     if (config.last_header_log_step >= config.header_log_step_frequency) {
//         write_header();
//         config.last_header_log_step = 0;
//     } else {
//         config.last_header_log_step += 1;
//     }
//     std::cout << std::setw(config.width) << step << " | " << std::setw(config.width);
//     for (long i = 0; i < orchestrator.log_names.size(); i++) {
//         double value = orchestrator.get_value(orchestrator.log_names[i]);
//         value = orchestrator.apply_modifier(orchestrator.log_names[i], value);
//         std::cout << std::setw(config.width) << std::scientific << std::setprecision(config.precision) << value;
//         if (i < orchestrator.log_names.size() - 1) {
//             std::cout << " | ";
//         }
//     }
//     std::cout << std::endl;
// }
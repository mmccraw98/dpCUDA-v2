#ifndef CONSOLE_LOG_IMPL_H
#define CONSOLE_LOG_IMPL_H

#include "console_log.h"

template <typename ParticleType, typename IntegratorType>
ConsoleLog<ParticleType, IntegratorType>::ConsoleLog(LogGroupConfig config, Orchestrator<ParticleType, IntegratorType>& orchestrator)
    : MacroLog<ParticleType, IntegratorType>(config, orchestrator) {
}

template <typename ParticleType, typename IntegratorType>
ConsoleLog<ParticleType, IntegratorType>::~ConsoleLog() {
}

template <typename ParticleType, typename IntegratorType>
void ConsoleLog<ParticleType, IntegratorType>::write_header() {
    std::ostringstream out;
    out << std::string(this->width * this->config.log_names.size() + (this->config.log_names.size() - 1), '_') << std::endl;
    for (int i = 0; i < this->config.log_names.size(); i++) {
        out << std::setw(this->width) << this->config.log_names[i];
        if (i < this->config.log_names.size() - 1) {
            out << this->delimiter;
        }
    }
    out << std::endl << std::string(this->width * this->config.log_names.size() + (this->config.log_names.size() - 1), '_') << std::endl;
    std::cout << out.str();
}

template <typename ParticleType, typename IntegratorType>
void ConsoleLog<ParticleType, IntegratorType>::log(long step) {
    if (this->last_header_log > this->header_log_freq) {
        this->write_header();
        this->last_header_log = 0;
    }
    this->last_header_log += 1;
    std::ostringstream out;
    for (int i = 0; i < this->config.log_names.size(); i++) {
        double value = this->orchestrator.template get_value<double>(this->unmodified_log_names[i], step);
        if (this->log_name_is_modified(this->config.log_names[i])) {
            std::string modifier = this->get_modifier(this->config.log_names[i]);
            value = this->orchestrator.apply_modifier(modifier, value);
        }
        out << std::setw(this->width) << std::scientific << std::setprecision(this->precision) << value;
        if (i < this->config.log_names.size() - 1) {
            out << this->delimiter;
        }
    }
    std::cout << out.str() << std::endl;
}


#endif /* CONSOLE_LOG_IMPL_H */
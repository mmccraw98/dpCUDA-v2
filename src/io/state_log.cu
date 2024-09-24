#include "../../include/io/state_log.h"
#include <thrust/host_vector.h>

StateLog::StateLog(LogGroupConfig config, Orchestrator& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix)
    : BaseLogGroup(config, orchestrator) : root_path(root_path), indexed_file_prefix(indexed_file_prefix) {
}

StateLog::~StateLog() {
}

void StateLog::log(long step) {
    for (int i = 0; i < config.log_names.size(); i++) {
        thrust::host_vector<double> value = orchestrator.get_vector_value(config.log_names[i]);
        std::vector<long> size = orchestrator.get_vector_size(config.log_names[i]);

        write_array_to_file(
            root_path / indexed_file_prefix + config.log_names[i] + extension,
            value,
            size[0],
            size[1],
            precision
        );
    }
}

StateLog StateLog::from_names_lin(Orchestrator& orchestrator, std::vector<std::string> log_names, long num_steps, long num_saves) {
    LogGroupConfig config = config_from_names_lin(log_names, num_steps, num_saves, "state");
    return StateLog(config, orchestrator);
}

StateLog StateLog::from_names_log(Orchestrator& orchestrator, std::vector<std::string> log_names, long num_steps, long num_saves, long min_save_decade) {
    LogGroupConfig config = config_from_names_log(log_names, num_steps, num_saves, min_save_decade, "state");
    return StateLog(config, orchestrator);
}

#include "../../include/io/state_log.h"
#include "../../include/io/utils.h"
#include <thrust/host_vector.h>

StateLog::StateLog(LogGroupConfig config, Orchestrator& orchestrator, const std::string& root_path, const std::string& indexed_file_prefix, const std::string& extension)
    : BaseLogGroup(config, orchestrator), root_path(root_path), indexed_file_prefix(indexed_file_prefix), extension(extension) {
}

StateLog::~StateLog() {
}

void StateLog::write_values(std::filesystem::path root_path) {
    for (int i = 0; i < config.log_names.size(); i++) {
        thrust::host_vector<double> value = orchestrator.get_vector_value<double>(config.log_names[i]);
        std::vector<long> size = orchestrator.get_vector_size(config.log_names[i]);
        std::filesystem::path file_path = root_path / (config.log_names[i] + extension);
        write_array_to_file(
            file_path.string(),
            value,
            size[0],
            size[1],
            precision
        );
    }
}

void StateLog::log(long step) {
    std::filesystem::path timestep_root_path = std::filesystem::path(root_path) / (indexed_file_prefix + std::to_string(step));
    make_dir(timestep_root_path.string(), true);
    write_values(timestep_root_path);
}

void StateLog::write_state() {
    write_values(root_path);
}
#include "../../include/io/state_log.h"
#include "../../include/io/io_utils.h"
#include <thrust/host_vector.h>

StateLog::StateLog(LogGroupConfigDict config, Orchestrator& orchestrator, const std::string& root, const std::string& indexed_file_prefix, const std::string& extension)
    : BaseLogGroup(config, orchestrator), root(root), indexed_file_prefix(indexed_file_prefix), extension(extension) {
    this->parallel = true;
}

StateLog::~StateLog() {
}

// need a key: value pair for each reorder array

void StateLog::gather_data(long step) {
    if (orchestrator.arrays_need_reordering) {
        reorder_index_data = orchestrator.get_reorder_index_data();
    }
    for (const auto& name : log_names) {
        ArrayData array_data = orchestrator.get_array_data(name);
        gathered_data[name] = array_data;
    }
}

void StateLog::write_values(const std::filesystem::path& root) {
    for (auto& [name, array_data] : gathered_data) {
        std::filesystem::path file_path = root / (name + extension);
        if (orchestrator.arrays_need_reordering && reorder_index_data.find(array_data.index_array_name) != reorder_index_data.end()) {
            reorder_array(array_data, reorder_index_data[array_data.index_array_name]);
        }
        write_array_data_to_file(file_path.string(), array_data, precision);
    }
}


void StateLog::log(long step) {
    std::filesystem::path timestep_root_path = std::filesystem::path(root) / (indexed_file_prefix + std::to_string(step));
    make_dir(timestep_root_path.string(), true);
    write_values(timestep_root_path);
}

void StateLog::write_state_to_path(const std::filesystem::path& path) {
    write_values(path);
}
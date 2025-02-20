#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/nve.h"
#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"
#include "../include/particles/factory.h"
#include "../include/particles/standard_configs.h"
#include "../include/integrator/adam.h"
#include "../include/routines/compression.h"
#include "../include/routines/initialization.h"
#include "../include/particles/factory.h"
#include "../include/io/io_utils.h"
#include "../include/io/arg_parsing.h"

int main(int argc, char** argv) {
    auto [particle, step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);
    std::filesystem::path input_dir = run_config["input_dir"].get<std::filesystem::path>();
    ConfigDict restart_config = state_config;
    restart_config["group_name"] = "restart";
    std::vector<ConfigDict> log_group_configs = {restart_config};
    IOManager dynamics_io_manager(log_group_configs, *particle, nullptr, input_dir, 20, false);
    dynamics_io_manager.log(step, true);
    return 0;
}
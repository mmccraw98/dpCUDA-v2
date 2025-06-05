#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/damped_nve.h"
#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"
#include "../include/particles/factory.h"
#include "../include/particles/standard_configs.h"
#include "../include/integrator/fire.h"
#include "../include/routines/compression.h"
#include "../include/routines/initialization.h"
#include "../include/particles/factory.h"
#include "../include/io/io_utils.h"
#include "../include/io/arg_parsing.h"

int main(int argc, char** argv) {
    auto [particle, compression_step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1, "restart")
    };
    IOManager io_manager(log_group_configs, *particle, nullptr, output_dir, 1, overwrite);
    io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    io_manager.log(0, true);
    return 0;
}
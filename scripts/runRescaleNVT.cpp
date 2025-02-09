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

    // assign the run config variables
    long rescale_freq = run_config["rescale_freq"].get<long>();
    long compression_freq = run_config["compression_freq"].get<long>();
    long num_steps = run_config["num_steps"].get<long>() + step;
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double temperature = run_config["temperature"].get<double>();
    double packing_fraction_increment = run_config["packing_fraction_increment"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    particle->setRandomVelocities(temperature);

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_dir, 20, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
        if (step % rescale_freq == 0) {
            particle->scaleVelocitiesToTemperature(temperature);
        }
        if (step % compression_freq == 0) {
            double phi = particle->getPackingFraction();
            particle->scaleToPackingFractionFull(phi + packing_fraction_increment);
        }
    }
    dynamics_io_manager.log(step, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
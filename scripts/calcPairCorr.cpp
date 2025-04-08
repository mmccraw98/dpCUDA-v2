#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/nve.h"
#include "../include/integrator/fire.h"
#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"
#include "../include/particles/factory.h"
#include "../include/particles/standard_configs.h"
#include "../include/routines/compression.h"
#include "../include/routines/initialization.h"
#include "../include/particles/factory.h"
#include "../include/io/io_utils.h"
#include "../include/io/arg_parsing.h"

int main(int argc, char** argv) {
    auto [particle, step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    // assign the run config variables
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    double avg_pe_tol = run_config["avg_pe_tol"];
    double phi_target = run_config["phi_target"];
    double temperature = run_config["temperature"];
    double final_temperature = run_config["final_temperature"];
    long num_equil_steps = run_config["num_equil_steps"];
    long num_steps = run_config["num_steps"];
    double compression_per_step = run_config["compression_per_step"];
    bool overwrite = true;
    double phi = particle->getPackingFraction();
    long num_compression_steps = static_cast<long>(std::round((phi_target - phi) / compression_per_step));

    // compress to the target packing fraction
    ConfigDict nve_config_dict = get_nve_config_dict(1e-2);
    NVE nve(*particle, nve_config_dict);
    std::vector<std::string> console_log_names = console_config["log_names"].get<std::vector<std::string>>();
    std::vector<ConfigDict> console_log_config = {config_from_names_lin(console_log_names, std::min(num_compression_steps, num_equil_steps), 10, "console")};
    IOManager compression_io_manager(console_log_config, *particle, &nve, output_dir, 1, overwrite);
    particle->setRandomVelocities(temperature);
    step = 0;
    while (step < num_compression_steps + 1) {
        particle->scaleToPackingFractionFull(phi + compression_per_step * step);
        particle->scaleVelocitiesToTemperature(temperature);
        nve.step();
        compression_io_manager.log(step);
        step++;
    }
    // allow to equilibrate at the target packing fraction
    step = 0;
    while (step < num_equil_steps) {
        particle->scaleVelocitiesToTemperature(temperature);
        nve.step();
        compression_io_manager.log(step);
        step++;
    }

    // run dynamics at specified temperature, saving the configuration periodically
    // if the temperature is 0, dynamics are still run, but the potential energy is minimized before the configuration is saved
    double dynamics_temperature = final_temperature;
    if (final_temperature == 0) {
        dynamics_temperature = temperature;
    }
    step = 0;
    long state_save_frequency = state_config["save_freq"].get<long>();
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, num_steps, "restart")
    };
    IOManager pair_corr_io_manager(log_group_configs, *particle, &nve, output_dir, 10, overwrite);
    pair_corr_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    while (step < num_steps) {
        particle->scaleVelocitiesToTemperature(dynamics_temperature);
        nve.step();
        if (final_temperature == 0 && step % state_save_frequency == 0) {
            minimizeFire(*particle, avg_pe_tol, 1e-16);
        }
        pair_corr_io_manager.log(step);
        if (final_temperature == 0 && step % state_save_frequency == 0) {  // need to re-add velocities since FIRE removes them
            particle->setRandomVelocities(dynamics_temperature);
        }
        step++;
    }
    if (final_temperature == 0) {
        minimizeFire(*particle, avg_pe_tol, 1e-16);
    }
    pair_corr_io_manager.log(step + 1, true);
    return 0;
}
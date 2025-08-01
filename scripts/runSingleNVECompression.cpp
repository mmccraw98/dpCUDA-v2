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
    long num_steps = run_config["num_steps"].get<long>();
    double phi_increment = run_config["phi_increment"].get<double>();
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double temperature = run_config["temperature"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool resume = run_config["resume"].get<bool>();
    bool overwrite = true;

    if (resume) {
        std::cout << "Resuming from: " << output_dir << std::endl;
        output_dir = run_config["input_dir"].get<std::filesystem::path>();
    }

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    
    if (!resume) {
        std::cout << "Starting compression and NVT relaxation run" << std::endl;
        particle->setRandomVelocities(temperature);
        double phi = particle->getPackingFraction();
        particle->scaleToPackingFractionFull(phi + phi_increment);
        phi = particle->getPackingFraction();
        particle->config["packing_fraction"] = phi;
        std::vector<ConfigDict> nvt_logger_configs = {console_config};
        IOManager nvt_io_manager(nvt_logger_configs, *particle, &nve, output_dir, 20, overwrite);
        long relaxation_step = 0;
        while (relaxation_step < num_steps / 10) {
            nve.step();
            nvt_io_manager.log(relaxation_step);
            relaxation_step++;
            if (relaxation_step % rescale_freq == 0) {
                particle->removeMeanVelocities();
                particle->scaleVelocitiesToTemperature(temperature);
            }
        }
        std::cout << "Starting NVE dynamics run" << std::endl;
    } else {
        std::cout << "Resuming NVE dynamics run" << std::endl;
    }

    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_dir, 20, !resume);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    auto start_time = std::chrono::high_resolution_clock::now();
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }
    dynamics_io_manager.log(step, true);
    std::cout << "Done with path: " << output_dir << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    return 0;
}
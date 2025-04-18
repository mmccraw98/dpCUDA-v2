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
    long num_steps = run_config["num_steps"].get<long>() + step;
    double phi_target = run_config["phi_target"].get<double>();
    double phi_increment = run_config["phi_increment"].get<double>();
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double temperature = run_config["temperature"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    particle->setRandomVelocities(temperature);

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };

    for (double phi = particle->getPackingFraction(); phi <= phi_target; phi += phi_increment) {
        // get the new output directory
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << phi;
        std::ostringstream oss_T;
        oss_T << std::scientific << std::setprecision(6) << temperature;
        std::filesystem::path sample_dir = output_dir / ("phi-" + oss.str()) / ("compression_T-" + oss_T.str());

        // clear the output directory
        std::filesystem::remove_all(sample_dir);

        particle->config["packing_fraction"] = phi;
        
        IOManager dynamics_io_manager(log_group_configs, *particle, &nve, sample_dir, 20, overwrite);
        dynamics_io_manager.write_params();
        run_config.save(sample_dir / "system" / "run_config.json");

        auto start_time = std::chrono::high_resolution_clock::now();

        step = 0;
        while (step < num_steps) {
            nve.step();
            dynamics_io_manager.log(step);
            step++;
            if (step % rescale_freq == 0) {
                particle->scaleVelocitiesToTemperature(temperature);
            }
        }
        dynamics_io_manager.log(step, true);
        std::cout << "Done with path: " << sample_dir << std::endl;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

        particle->scaleToPackingFractionFull(phi + phi_increment);
    }

    return 0;
}
#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/nve.h"
#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"
#include "../include/particles/factory.h"
#include "../include/particles/standard_configs.h"
#include "../include/integrator/adam.h"
#include "../include/integrator/fire.h"
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
    bool overwrite = true;

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_dir, 20, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    auto start_time = std::chrono::high_resolution_clock::now();

    double avg_pe_tol = 1e-16;
    double avg_pe_diff_tol = 1e-16;
    double phi_low = -1;
    double phi_high = -1;
    double phi_tolerance = 1e-6;  // difference between phi_high and phi_low to declare convergence
    double phi = particle->getPackingFraction();
    double next_phi = phi + phi_increment;

    long iteration = 0;
    while (true) {
        iteration++;
        particle->setRandomVelocities(temperature);
        particle->scaleToPackingFractionFull(next_phi);
        phi = particle->getPackingFraction();
        while (step < num_steps * iteration) {
            nve.step();
            if (step % rescale_freq == 0) {
                particle->scaleVelocitiesToTemperature(temperature);
            }
            dynamics_io_manager.log(step);
            step++;
        }
        minimizeFire(*particle, avg_pe_tol, avg_pe_diff_tol, 1e6);
        double avg_pe = particle->totalPotentialEnergy() / particle->n_particles;

        if (avg_pe > avg_pe_tol) {
            phi_high = phi;
            next_phi = (phi_high + phi_low) / 2.0;
        } else {
            phi_low = phi;
            if (phi_high > 0) {
                next_phi = (phi_high + phi_low) / 2.0;
            } else {
                next_phi += phi_increment;
            }
        }
        if (std::abs(phi_high / phi_low - 1) < phi_tolerance && phi_high > 0 && phi_low > 0) {
            break;
        }
    }

    std::cout << phi_high << " " << phi_low << std::endl;
    std::cout << std::abs(phi_high / phi_low - 1) << std::endl;

    dynamics_io_manager.log(step, true);
    std::cout << "Done with path: " << output_dir << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    return 0;
}
#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/damped_nve.h"
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
    auto [particle, compression_step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    // assign the run config variables
    long num_compression_steps = run_config["num_compression_steps"].get<long>();
    long num_dynamics_steps = run_config["num_dynamics_steps"].get<long>();
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double damping_coefficient = run_config["damping_coefficient"].get<double>();
    double avg_ke_target = run_config["avg_ke_target"].get<double>();
    double avg_pe_target = run_config["avg_pe_target"].get<double>();
    double compression_step_increment = run_config["compression_step_increment"].get<double>();
    double min_compression_step_increment = run_config["min_compression_step_increment"].get<double>();
    double pe_target_fraction = run_config["pe_target_fraction"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;


    long pe_hist_size = 1000;
    std::vector<double> pe_hist(pe_hist_size, 0.0);
    long pe_hist_index = 0;

    long last_decompression_step = 0;
    double max_compression_step_increment = compression_step_increment;

    long n_particles = particle->n_particles;

    ConfigDict damped_nve_config = get_damped_nve_config_dict(dt_dimless * particle->getTimeUnit() * particle->getGeometryScale(), damping_coefficient);
    DampedNVE damped_nve(*particle, damped_nve_config);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, &damped_nve, output_dir, 20, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");



    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // compression loop
    while (compression_step < num_compression_steps) {
        damped_nve.step();
        // log the potential energy
        pe_hist_index = (pe_hist_index + 1) % pe_hist_size;
        double pe = particle->totalPotentialEnergy() / n_particles;
        pe_hist[pe_hist_index] = pe;
        // calculate the potential energy fluctuation
        if (pe_hist_index == pe_hist_size - 1) {
            double pe_avg = std::accumulate(pe_hist.begin(), pe_hist.end(), 0.0) / pe_hist_size;
            double pe_fluct = 0.0;
            if (pe_avg != 0.0) {
                for (long i = 0; i < pe_hist_size; i++) {
                    pe_fluct += (pe_hist[i] - pe_avg) * (pe_hist[i] - pe_avg);
                }
                pe_fluct = sqrt(pe_fluct / pe_hist_size) / pe_avg;
            }
            if (pe_fluct < avg_pe_target || pe < avg_pe_target) {
                // compress
                particle->scaleToPackingFractionFull(particle->getPackingFraction() + compression_step_increment);
            }
        }
        dynamics_io_manager.log(compression_step);
        compression_step++;
    }
    dynamics_io_manager.log(compression_step, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

















    // ----------------------------------------------------------------------
    // OLD COMPRESSION LOOP
    // ----------------------------------------------------------------------

    // // start the timer
    // auto start_time = std::chrono::high_resolution_clock::now();

    // // compression loop
    // while (compression_step < num_compression_steps) {
        
    //     // run dynamics until kinetic energy per particle is below a set value
    //     long dynamics_step = 0;
    //     while (dynamics_step < num_dynamics_steps) {
    //         damped_nve.step();
    //         particle->calculateKineticEnergy();
    //         double avg_ke = particle->totalKineticEnergy() / n_particles;
    //         if (avg_ke < avg_ke_target) {
    //             break;
    //         }
    //         dynamics_step++;
    //     }
    //     dynamics_io_manager.log(compression_step);

    //     // if the potential energy per particle is below a set value, compress, else decompress
    //     double avg_pe = particle->totalPotentialEnergy() / n_particles;
    //     double sign = 1.0;
    //     if (avg_pe > avg_pe_target) {
    //         sign = -1.0;
    //         last_decompression_step = compression_step;
    //         if (compression_step_increment / 2.0 > min_compression_step_increment) {
    //             compression_step_increment /= 2.0;
    //             std::cout << "SHRINKING STEP SIZE" << std::endl;
    //         }
    //     }
    //     // if the potential energy is just slightly below the target, it is done (only if the step size is sufficiently small to avoid early exits)
    //     else if (avg_pe > pe_target_fraction * avg_pe_target && compression_step_increment < 2.0 * min_compression_step_increment) {
    //         break;
    //     }
    //     if (compression_step - last_decompression_step > 100) {
    //         std::cout << "EXPANDING STEP SIZE" << std::endl;
    //         compression_step_increment = std::min(compression_step_increment * 2.0, max_compression_step_increment);
    //     }
    //     particle->scaleToPackingFractionFull(particle->getPackingFraction() + compression_step_increment * sign);
    //     compression_step++;
    // }
    // dynamics_io_manager.log(compression_step, true);

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    // std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
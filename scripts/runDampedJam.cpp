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

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();

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

    double e_c = particle->getEnergyScale("c");
    avg_pe_target /= e_c;

    long last_decompression_step = 0;
    double max_compression_step_increment = compression_step_increment;

    long n_particles = particle->n_particles;

    ConfigDict damped_nve_config = get_damped_nve_config_dict(dt_dimless * particle->getTimeUnit() * particle->getGeometryScale(), damping_coefficient);
    DampedNVE damped_nve(*particle, damped_nve_config);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient"};
        pair_names.insert(pair_names.end(), rb_pair_names.begin(), rb_pair_names.end());
    }
    init_names.insert(init_names.end(), pair_names.begin(), pair_names.end());
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, &damped_nve, output_dir, 20, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");


    // ----------------------------------------------------------------------
    // NEW COMPRESSION LOOP
    // ----------------------------------------------------------------------


    // // start the timer
    // auto start_time = std::chrono::high_resolution_clock::now();

    // // compression loop
    // long pe_hist_size = 1000;
    // std::vector<double> pe_hist(pe_hist_size, 0.0);
    // long pe_hist_index = 0;
    // double last_sign = 1.0;
    // double sign = 1.0;
    // long compression_step_index = 0;
    // while (compression_step < num_compression_steps) {
    //     damped_nve.step();
    //     // log the potential energy
    //     pe_hist_index = (pe_hist_index + 1) % pe_hist_size;
    //     double pe = particle->totalPotentialEnergy() / n_particles;
    //     pe_hist[pe_hist_index] = pe;
    //     // calculate the potential energy fluctuation
    //     if (pe_hist_index == pe_hist_size - 1) {
    //         double pe_avg = std::accumulate(pe_hist.begin(), pe_hist.end(), 0.0) / pe_hist_size;
    //         double pe_fluct = 0.0;
    //         if (pe_avg != 0.0) {
    //             for (long i = 0; i < pe_hist_size; i++) {
    //                 pe_fluct += (pe_hist[i] - pe_avg) * (pe_hist[i] - pe_avg);
    //             }
    //             pe_fluct = sqrt(pe_fluct / pe_hist_size) / pe_avg;
    //         }
    //         if (pe_fluct < avg_pe_target || pe < avg_pe_target) {
    //             compression_step_index++;
    //             if (pe < avg_pe_target) {
    //                 // compress
    //                 sign = 1.0;
    //             }
    //             else if (pe < avg_pe_target * pe_target_fraction && compression_step_increment < 2.0 * min_compression_step_increment) {
    //                 // done
    //                 break;
    //             }
    //             else {
    //                 // decompress
    //                 sign = -1.0;
    //                 last_decompression_step = compression_step;
    //             }
    //             if (last_sign != sign) {
    //                 compression_step_increment = std::max(compression_step_increment / 2.0, min_compression_step_increment);
    //             }
    //             if (compression_step_index - last_decompression_step > 100) {
    //                 last_decompression_step = compression_step_index;
    //                 compression_step_increment = std::min(compression_step_increment * 2.0, max_compression_step_increment);
    //             }
    //             particle->scaleToPackingFractionFull(particle->getPackingFraction() + compression_step_increment * sign);
    //             particle->removeMeanVelocities();

    //             last_sign = sign;
    //         }
    //     }
    //     dynamics_io_manager.log(compression_step);
    //     compression_step++;
    // }
    // dynamics_io_manager.log(compression_step, true);

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    // std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

















    // ----------------------------------------------------------------------
    // OLD COMPRESSION LOOP
    // ----------------------------------------------------------------------

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    double last_sign = 1.0;

    // compression loop
    while (compression_step < num_compression_steps) {
        
        // run dynamics until kinetic energy per particle is below a set value
        long dynamics_step = 0;
        while (dynamics_step < num_dynamics_steps) {
            damped_nve.step();
            particle->calculateKineticEnergy();
            double avg_ke = particle->totalKineticEnergy() / n_particles;
            if (avg_ke < avg_ke_target) {
                break;
            }
            dynamics_step++;
        }
        dynamics_io_manager.log(compression_step);

        double avg_pe = particle->totalPotentialEnergy() / n_particles;
        double sign = 1.0;
        // if the potential energy is just slightly below the target, it is done (only if the step size is sufficiently small to avoid early exits)
        if (avg_pe > avg_pe_target && avg_pe < avg_pe_target * pe_target_fraction && compression_step_increment < 2.0 * min_compression_step_increment) {
            break;
        }
        // if the potential energy per particle is below a set value, compress, else decompress
        if (avg_pe > avg_pe_target) {
            sign = -1.0;
            last_decompression_step = compression_step;
        }
        // if we just switched between compressing/decompressing, shrink the step size
        if (last_sign != sign) {
            // std::cout << "SHRINKING STEP SIZE" << std::endl;
            compression_step_increment = std::max(compression_step_increment / 2.0, min_compression_step_increment);
        }
        // if we haven't had to decompress for a while, expand the step size slowly
        if (compression_step - last_decompression_step > 100) {
            last_decompression_step = compression_step;
            // std::cout << "EXPANDING STEP SIZE" << std::endl;
            compression_step_increment = std::min(compression_step_increment * 2.0, max_compression_step_increment);
        }
        particle->scaleToPackingFractionFull(particle->getPackingFraction() + compression_step_increment * sign);
        particle->removeMeanVelocities();
        compression_step++;
        last_sign = sign;
    }
    dynamics_io_manager.log(compression_step, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
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

// Appropriated from Arthur MacKeith's onboarding pseudo code

// This code works by setting high and low density bounds and then using a binary search protocol to find a jammed state between them
// The system is compressed and its potential energy per particle (PE/P) is monitored
// If the PE/P is below a tolerance, the configuration is saved as the lower bound and the system is compressed
// If the PE/P is above a tolerance, the upper bound is set and the system is reverted to the lower bound
// The system is then compressed from the lower bound to the halfway point between the lower and upper bounds
// The process is repeated
// If the new configuration is below the tolerance, a new lower bound is set
// The new lower bound no longer has the same upper bound so the entire process is repeated
// Then, any jammed state will be associated with a given upper and lower bound
// In practice, this leads to oscillations and early exits
// To avoid this, we introduce a slight modification
// Here, we require that a jammed state has to have been associated with an upper bound that exceeds some critical PE/P threshold
// This ensures that the system is stable under compressions

// NOTE:
// if the initial packing fraction is too low and a cell-list is used, the cells may be too small to contain any particles
// and the cell-list may fail once compressed to higher densities

int main(int argc, char** argv) {
    auto [particle, compression_step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();

    // assign the run config variables
    long num_compression_steps = run_config["num_compression_steps"].get<long>();
    double compression_rate = run_config["compression_rate"].get<double>();
    double avg_pe_tolerance_low = run_config["avg_pe_tolerance_low"].get<double>();
    double avg_pe_tolerance_high = run_config["avg_pe_tolerance_high"].get<double>();
    double length_tolerance = run_config["length_tolerance"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    // we have to start with no contacts
    long n_particles = particle->n_particles;
    while (true) {
        minimizeFire(*particle, 1e-16, 0);  // need to start with no contacts
        particle->calculateStressTensor();
        double pe_per_particle = particle->totalPotentialEnergy() / n_particles;
        if (pe_per_particle < avg_pe_tolerance_low) {
            break;
        }
        particle->scaleBox(1.01);
    }


    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs", "contact_counts", "hessian_pairs_xx", "hessian_pairs_xy", "hessian_pairs_yy"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient", "pair_vertex_overlaps"};
        pair_names.insert(pair_names.end(), rb_pair_names.begin(), rb_pair_names.end());
    }
    init_names.insert(init_names.end(), pair_names.begin(), pair_names.end());
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1, "restart")
    };
    IOManager io_manager(log_group_configs, *particle, nullptr, output_dir, 1, overwrite);
    io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // compression loop
    double critical_pe_per_particle = 1e-8;  // if we reach this, we have reached the critical point
    bool critical_point_reached = false;
    double pe_diff_target_critical = avg_pe_tolerance_low * 1e-5;  // these need to be very small
    double pe_diff_target = avg_pe_tolerance_low * 1e-8;
    // after we reach the critical point, we are no longer allowed to increase L_high

    double L_low = -1, L_high = -1, L_last = -1;
    double eps = 1 - compression_rate;
    while (compression_step < num_compression_steps) {
        // minimizeFire(*particle, pressure_tolerance_low / 2, 0);
        minimizeFire(*particle, 0.1, 1e-2, avg_pe_tolerance_low * 0.9, pe_diff_target, 1e6, 1e4);
        double pe_per_particle = particle->totalPotentialEnergy() / n_particles;
        particle->countContacts();
        long num_contacts = particle->getContactCount();
        double L = std::sqrt(particle->getBoxArea());
        double L_prior = L;
        io_manager.log(compression_step, true);

        if (pe_per_particle > critical_pe_per_particle) {
            critical_point_reached = true;
            pe_diff_target = pe_diff_target_critical;
        }

        if (pe_per_particle < avg_pe_tolerance_low) {  // unjammed state
            L_low = L;
            // SAVE: set last state to current state
            // io_manager.write_restart_file(compression_step, "last_state");
            std::cout << "Saving last state" << std::endl;
            particle->setLastState();
            L_last = L;
            if (L_high != -1) {  // searching within L_hi and L_lo
                L = (L_high + L_low) / 2;
                if (!critical_point_reached) {
                    L_high = -1;  // unset L_high
                }
            } else {  // not searching within L_hi and L_lo
                L = L * compression_rate;
            }
        } else {  // jammed state or final state
            if (pe_per_particle > avg_pe_tolerance_high) {  // jammed state
                L_high = L;
                // REVERT: set current state to last state
                // particle->load(output_dir, "last_state");
                std::cout << "Reverting to last state" << std::endl;
                particle->revertToLastState();
                L_prior = L_last;
                L = (L_high + L_low) / 2;
            } else if (num_contacts > 2 * n_particles && critical_point_reached) {  // final state has to have come from a chain which included a critical point
                break;  // added contact criteria to avoid early termination
            }
        }
        if (L_low > 0 && L_high > 0 && std::abs(L_low / L_high - 1) < length_tolerance) {
            std::cout << "Error: couldnt find jammed state" << std::endl;
            break;
        }
        // scale box to L and bring positions with it
        particle->scaleBox(L / L_prior);
        compression_step++;
    }
    io_manager.log(compression_step + 1, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
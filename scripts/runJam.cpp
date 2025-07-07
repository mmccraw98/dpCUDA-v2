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

static bool is_stable_configuration(Particle& particle, double delta_phi, double avg_pe_tolerance, double avg_pe_diff_tolerance) {
    double initial_pe = particle.totalPotentialEnergy() / particle.n_particles;
    particle.setLastState();
    particle.scaleToPackingFractionFull(particle.getPackingFraction() + delta_phi);
    minimizeFire(particle, avg_pe_tolerance, avg_pe_diff_tolerance);
    double final_pe = particle.totalPotentialEnergy() / particle.n_particles;
    particle.revertToLastState();
    return final_pe > 1e-1 * std::pow(delta_phi, particle.n_c);
}

int main(int argc, char** argv) {
    auto [particle, compression_step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();

    // assign the run config variables
    long num_compression_steps = run_config["num_compression_steps"].get<long>();
    double compression_increment = run_config["compression_increment"].get<double>();
    double avg_pe_tolerance = run_config["avg_pe_tolerance"].get<double>();
    double avg_pe_diff_tolerance = run_config["avg_pe_diff_tolerance"].get<double>();
    double phi_tolerance = run_config["phi_tolerance"].get<double>();
    double delta_phi_stability = run_config["delta_phi_stability"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    // we have to start with no contacts
    long n_particles = particle->n_particles;
    while (true) {
        minimizeFire(*particle, avg_pe_tolerance, avg_pe_diff_tolerance);  // need to start with no contacts
        double pe_per_particle = particle->totalPotentialEnergy() / n_particles;
        if (pe_per_particle < avg_pe_tolerance) {
            break;
        }
        particle->scaleBox(1.1);
    }


    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs", "contact_counts", "hessian_pairs_xx", "hessian_pairs_xy", "hessian_pairs_yx", "hessian_pairs_yy", "hessian_ii_xx", "hessian_ii_xy", "hessian_ii_yx", "hessian_ii_yy"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient", "pair_vertex_overlaps", "hessian_pairs_xt", "hessian_pairs_yt", "hessian_pairs_tt", "hessian_pairs_tx", "hessian_pairs_ty", "hessian_ii_xt", "hessian_ii_yt", "hessian_ii_tt", "hessian_ii_tx", "hessian_ii_ty", "area"};
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

    particle->setLastState();
    double phi_low  = particle->getPackingFraction();
    double phi;
    double phi_high = -1.0;
    while (compression_step < num_compression_steps) {
        phi = particle->getPackingFraction();
        minimizeFire(*particle, avg_pe_tolerance, avg_pe_diff_tolerance, 1e6);
        double pe_per_particle = particle->totalPotentialEnergy() / n_particles;
        particle->countContacts();
        long z_per_particle = particle->getContactCount() / n_particles;
        if (pe_per_particle > avg_pe_tolerance) {  // jammed
            phi_high = phi;
            phi = (phi_high + phi_low) / 2.0;
            particle->revertToLastState();
        } else {  // unjammed
            particle->setLastState();
            io_manager.log(compression_step, true);  // save a smooth trajectory
            phi_low = phi;
            if (phi_high > 0) {
                phi = (phi_high + phi_low) / 2.0;
            } else {
                phi += compression_increment;
            }
        }
        if (std::abs(phi_high / phi_low - 1) < phi_tolerance) {  // converged
            bool is_stable = is_stable_configuration(*particle, delta_phi_stability, avg_pe_tolerance, avg_pe_diff_tolerance);
            if (is_stable) {
                break;
            }
            std::cout << "unstable" << std::endl;
            phi_low = phi;
            phi_high = -1.0;
            phi = particle->getPackingFraction();
            particle->setLastState();
        }
        particle->scaleToPackingFractionFull(phi);
        compression_step++;
    }
    io_manager.log(compression_step + 1, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
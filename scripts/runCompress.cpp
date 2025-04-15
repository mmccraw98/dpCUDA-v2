#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/adam.h"
#include "../include/integrator/fire.h"
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
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    double delta_phi = run_config["delta_phi"].get<double>();
    double avg_pe_tolerance = run_config["avg_pe_tolerance"].get<double>();
    double avg_pe_diff_tolerance = run_config["avg_pe_diff_tolerance"].get<double>();
    bool overwrite = true;

    std::string particle_type = particle->config["particle_type"].get<std::string>();
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs", "contact_counts", "hessian_pairs_xx", "hessian_pairs_xy", "hessian_pairs_yx", "hessian_pairs_yy", "hessian_ii_xx", "hessian_ii_xy", "hessian_ii_yx", "hessian_ii_yy"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient", "pair_vertex_overlaps", "hessian_pairs_xt", "hessian_pairs_yt", "hessian_pairs_tt", "hessian_pairs_tx", "hessian_pairs_ty", "hessian_ii_xt", "hessian_ii_yt", "hessian_ii_tt", "hessian_ii_tx", "hessian_ii_ty"};
        pair_names.insert(pair_names.end(), rb_pair_names.begin(), rb_pair_names.end());
    }
    init_names.insert(init_names.end(), pair_names.begin(), pair_names.end());
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, nullptr, output_dir, 1, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    dynamics_io_manager.log(0, true);

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // compress the particle
    double phi = particle->getPackingFraction();
    particle->scaleToPackingFractionFull(phi + delta_phi);

    // minimizeAdam(*particle);

    minimizeFire(*particle, avg_pe_tolerance, avg_pe_diff_tolerance);
    dynamics_io_manager.log(1, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
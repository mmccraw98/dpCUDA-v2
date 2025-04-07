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
    long num_steps = run_config["num_steps"];
    long num_equil_steps = run_config["num_equil_steps"];
    bool overwrite = true;

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs", "contact_counts"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient", "pair_vertex_overlaps"};
        pair_names.insert(pair_names.end(), rb_pair_names.begin(), rb_pair_names.end());
    }
    init_names.insert(init_names.end(), pair_names.begin(), pair_names.end());
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, num_steps * 2, "restart")
    };
    ConfigDict nve_config_dict = get_nve_config_dict(1e-2);
    NVE nve(*particle, nve_config_dict);
    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_dir, 1, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    double phi = particle->getPackingFraction();
    particle->setRandomVelocities(temperature);
    double compression_per_step = (phi_target - phi) / static_cast<double>(num_steps);
    while (step < num_steps + 1) {
        particle->scaleToPackingFractionFull(phi + compression_per_step * step);
        particle->scaleVelocitiesToTemperature(temperature);
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }
    while (step < num_steps + 1 + num_equil_steps) {
        particle->scaleVelocitiesToTemperature(temperature);
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }


    if (final_temperature != temperature && final_temperature != 0) {
        while (step < num_steps + 1 + num_equil_steps * 2) {
            particle->scaleVelocitiesToTemperature(final_temperature);
            nve.step();
            dynamics_io_manager.log(step);
            step++;
        }
    } else if (final_temperature == 0) {
        minimizeFire(*particle, avg_pe_tol, 1e-16);
    }

    dynamics_io_manager.log(0, true);
    return 0;
}
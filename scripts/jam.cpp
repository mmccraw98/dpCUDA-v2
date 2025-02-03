#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/particles/base/config.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

#include "../include/particles/standard_configs.h"

#include "../include/integrator/adam.h"

#include "../include/routines/compression.h"

int main() {

    long n_particles = 32;
    double packing_fraction = 0.01;
    std::string particle_type = "RigidBumpy";
    // std::string particle_type = "Disk";
    long save_every_N_steps = 1e3;
    long num_compression_steps = 1e4;
    bool overwrite = true;
    // std::string output_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/rb/1/jamming/";
    // std::string base_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/disk/2/";
    // std::string base_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/disk/3/";
    std::string base_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/rb/4/";
    std::string output_path = base_path + "jamming/";

    std::vector<std::string> init_names = {"radii", "masses", "positions", "velocities", "forces", "box_size", "particle_index", "static_particle_index"};
    std::vector<std::string> state_names = {"positions", "velocities", "box_size", "particle_index", "static_particle_index"};

    ConfigDict particle_config;
    if (particle_type == "Disk") {
        particle_config = get_standard_disk_config(n_particles, packing_fraction);
    } else if (particle_type == "RigidBumpy") {
        particle_config = get_standard_rigid_bumpy_config(n_particles, packing_fraction);
        std::vector<std::string> additional_log_names = {"angles", "vertex_positions", "angular_velocities", "torques", "vertex_forces", "static_vertex_index", "particle_start_index", "vertex_particle_index", "num_vertices_in_particle"};
        for (const auto& name : additional_log_names) {
            init_names.push_back(name);
            state_names.push_back(name);
        }
    } else {
        throw std::runtime_error("Invalid particle type: " + particle_type);
    }

    auto particle = createParticle(particle_config);

    AdamConfigDict adam_config;
    adam_config["alpha"] = 1e-4;
    adam_config["beta1"] = 0.9;
    adam_config["beta2"] = 0.999;
    adam_config["epsilon"] = 1e-8;
    Adam adam(*particle, adam_config);

    std::vector<LogGroupConfigDict> log_group_configs = {
        // config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e2, "console"),
        // config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e2, "console"),
        config_from_names_lin_everyN({"step", "PE", "phi"}, save_every_N_steps, "energy"),
        config_from_names(init_names, "init"),
        config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
    };
    IOManager io_manager(log_group_configs, *particle, &adam, output_path, 1, overwrite);
    io_manager.write_params();

    long start_step = 0;
    long num_adam_steps = 1e5;
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    double packing_fraction_increment = 1e-4;  // 1e-4 for 1024, 1e-5 for 64
    double min_packing_fraction_increment = packing_fraction_increment * 1e-2;

    double overcompression_amount = 0.005;


    // jam_adam(*particle, adam, io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, min_packing_fraction_increment, 1.01, start_step);

    // compress_to_phi_adam(*particle, adam, io_manager, 1e5, num_adam_steps, avg_pe_target, avg_pe_diff_target, overcompression_amount, num_compression_steps);









    long num_iterations = 10;
    long sub_num_steps = static_cast<long>(num_compression_steps / num_iterations);
    for (long i = 0; i < num_iterations; i++) {
        double dt_dimless = 1e-3;
        NVEConfigDict nve_config_dict;
        nve_config_dict["dt"] = dt_dimless * particle->getTimeUnit();
        NVE nve(*particle, nve_config_dict);
        particle->setRandomVelocities(1e-4);

        long dynamic_step = 0;
        while (dynamic_step < sub_num_steps) {
            nve.step();
            dynamic_step++;
        }

        jam_adam(*particle, adam, io_manager, sub_num_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, min_packing_fraction_increment, 1.01, start_step);
        start_step += sub_num_steps;
    }

    compress_to_phi_adam(*particle, adam, io_manager, 1e5, num_adam_steps, avg_pe_target, avg_pe_diff_target, overcompression_amount, num_compression_steps);














    std::vector<double> temperatures = {1e-6, 1e-5, 1e-4};
    for (double temperature : temperatures) {
        output_path = base_path + "dynamics-T-" + std::to_string(temperature) + "/";
        save_every_N_steps = 1e3;
        long num_steps = 1e6;

        double dt_dimless = 1e-3;
        NVEConfigDict nve_config_dict;
        nve_config_dict["dt"] = dt_dimless * particle->getTimeUnit();
        NVE nve(*particle, nve_config_dict);

        init_names = {"radii", "masses", "positions", "velocities", "forces", "box_size"};
        state_names = {"positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j"};
        particle_type = particle->getConfig()["type_name"];
        if (particle_type == "RigidBumpy") {
            std::vector<std::string> additional_log_names = {"angles", "vertex_positions", "angular_velocities", "torques", "vertex_forces", "static_vertex_index", "particle_start_index", "vertex_particle_index", "num_vertices_in_particle"};
            for (const auto& name : additional_log_names) {
                init_names.push_back(name);
                state_names.push_back(name);
            }
        }
        log_group_configs = {
            config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
            config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
            config_from_names(init_names, "init"),
            config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
        };

        particle->setRandomVelocities(temperature);

        IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_path, 1, overwrite);
        dynamics_io_manager.write_params();

        long step = 0;
        while (step < num_steps) {
            nve.step();
            dynamics_io_manager.log(step);
            step++;
        }
    }

  

    return 0;
}
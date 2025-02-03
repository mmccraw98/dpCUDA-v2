#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/particles/base/config.h"

#include "../include/integrator/nve.h"
#include "../include/integrator/adam.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

#include "../include/routines/compression.h"


int main() {
    double temperature = 1e-9;
    // std::string input_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/rb/1/jamming/";
    // std::string output_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/rb/1/dynamics-T-" + std::to_string(temperature) + "/";

    std::string input_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/disk/2/jamming/";
    std::string output_path = "/home/mmccraw/dev/data/25-02-01/effective-potential/disk/2/dynamics-T-" + std::to_string(temperature) + "/";

    bool overwrite = true;
    long save_every_N_steps = 1e3;
    long num_steps = 1e6;

    long trajectory_frame = -1;
    std::filesystem::path input_path_obj(input_path);
    auto [particle, frameNumber, sysPath, trajPath] = loadParticleFromRoot(input_path_obj, trajectory_frame);

    double dt_dimless = 1e-2;
    NVEConfigDict nve_config_dict;
    nve_config_dict["dt"] = dt_dimless * particle->getTimeUnit();
    NVE nve(*particle, nve_config_dict);

    AdamConfigDict adam_config;
    adam_config["alpha"] = 1e-4;
    adam_config["beta1"] = 0.9;
    adam_config["beta2"] = 0.999;
    adam_config["epsilon"] = 1e-8;
    Adam adam(*particle, adam_config);

    std::vector<std::string> init_names = {"radii", "masses", "positions", "velocities", "forces", "box_size"};
    std::vector<std::string> state_names = {"positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j"};
    std::string particle_type = particle->getConfig()["type_name"];
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> additional_log_names = {"angles", "vertex_positions", "angular_velocities", "torques", "vertex_forces", "static_vertex_index", "particle_start_index", "vertex_particle_index", "num_vertices_in_particle"};
        for (const auto& name : additional_log_names) {
            init_names.push_back(name);
            state_names.push_back(name);
        }
    }
    std::vector<LogGroupConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
        config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        config_from_names(init_names, "init"),
        config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
    };


    // IOManager io_manager(log_group_configs, *particle, &adam, output_path, 1, overwrite);
    // io_manager.write_params();

    long start_step = 0;
    long num_adam_steps = 1e5;
    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    double packing_fraction_increment = 1e-6;
    double min_packing_fraction_increment = packing_fraction_increment * 1e-3;
    long num_compression_steps = 1e6;

    double packing_fraction;
    packing_fraction = particle->getPackingFraction();
    std::cout << "packing_fraction: " << packing_fraction << std::endl;

    // jam_adam(*particle, adam, io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, min_packing_fraction_increment, 1.0001, start_step);

    ConfigDict config = particle->getConfig();
    particle->setupNeighbors(config);
    particle->initNeighborList();
    particle->calculateForces();  // make sure forces are calculated before the integration
    double force_balance = particle->getForceBalance();
    if (force_balance / particle->n_particles / particle->e_c > 1e-14) {
        std::cout << "WARNING: Particle::setupNeighbors: Force balance is "
                  << force_balance << ", there will be an error!\n";
    }

    auto frame_path_and_frame = get_trajectory_frame_path(trajPath, "t", frameNumber);
    std::filesystem::path frame_path = std::get<0>(frame_path_and_frame);
    particle->loadDataFromPath(frame_path, ".dat");

    particle->setRandomVelocities(temperature);

    IOManager io_manager(log_group_configs, *particle, &nve, output_path, 1, overwrite);
    io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.step();
        io_manager.log(step);
        step++;
    }

    // NEED TO CALCULATE ALL THE PARTICLE FORCES AND DISTANCES FOR ALL PAIRS
    // SHOULD ALSO CALCULATE PAIR ANGLES (I.E. ANGLE I ANGLE J)

    return 0;
}
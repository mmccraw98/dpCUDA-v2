#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/particles/base/config.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

int main() {

    // BidispersityConfigDict dispersity_config;
    // dispersity_config["size_ratio"] = 1.4;
    // dispersity_config["count_ratio"] = 0.5;

    // PointNeighborListConfigDict neighbor_list_config;
    // neighbor_list_config["neighbor_cutoff_multiplier"] = 1.5;
    // neighbor_list_config["neighbor_displacement_multiplier"] = 0.2;
    // neighbor_list_config["num_particles_per_cell"] = 8.0;
    // neighbor_list_config["cell_displacement_multiplier"] = 0.5;
    // neighbor_list_config["neighbor_list_update_method"] = "cell";

    // DiskParticleConfigDict particle_config;
    // particle_config["dispersity_config"] = dispersity_config.to_nlohmann_json();
    // particle_config["neighbor_list_config"] = neighbor_list_config.to_nlohmann_json();
    // particle_config["packing_fraction"] = 0.8;
    // particle_config["n_particles"] = 64;
    // particle_config["e_c"] = 1.0;
    // particle_config["n_c"] = 2.0;
    // particle_config["particle_dim_block"] = 256;
    // particle_config["seed"] = 0;
    // particle_config["mass"] = 1.0;

    // auto particle = createParticle(particle_config);

    // // std::cout << particle_config.to_nlohmann_json().dump(4) << std::endl;
    // // particle_config.to_json("/home/mmccraw/dev/data/24-11-08/test/file.json");
    // // DiskParticleConfigDict config2;
    // // config2.from_json("/home/mmccraw/dev/data/24-11-08/test/file.json");
    // // std::cout << config2.to_nlohmann_json().dump(4) << std::endl;

    // // make a particle factory so that given a config, you create any particle type

    // // particle.load(path, frame) - frame = -1 is the last frame

    // long num_steps = 1e3;
    // long save_every_N_steps = 1e2;
    // double dynamics_temperature = 1e-4;
    // double dt_dimless = 1e-2;
    // bool overwrite = true;
    // std::string path = "/home/mmccraw/dev/data/24-11-08/test-data";

    // NVEConfigDict nve_config;
    // nve_config["dt"] = dt_dimless * particle->getTimeUnit();

    // particle->setRandomVelocities(dynamics_temperature);
    // NVE nve(*particle, nve_config);

    // std::vector<LogGroupConfigDict> log_group_configs = {
    //     config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T"}, 1e2, "console"),  // logs to the console
    //     config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    //     config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
    //     config_from_names_lin_everyN({"positions", "forces", "velocities", "force_pairs", "distance_pairs", "num_neighbors", "neighbor_list", "static_particle_index", "pair_ids"}, save_every_N_steps, "state"),
    // };
    // IOManager io_manager(log_group_configs, *particle, &nve, path, 1, overwrite);
    // io_manager.write_params();
    // long step = 0;
    // while (step < num_steps) {
    //     nve.step();
    //     io_manager.log(step);
    //     step++;
    // }





    std::string root = "/home/mmccraw/dev/data/24-11-08/test-data";
    std::string output_path = "/home/mmccraw/dev/data/24-11-08/test-data-2";

    long trajectory_frame = -1;

    std::filesystem::path root_path(root);

    // // get the paths to all the log_config files
    // std::vector<std::filesystem::path> log_config_paths;
    // for (const auto& entry : std::filesystem::directory_iterator(system_path)) {
    //     if (entry.path().extension() == ".json" && entry.path().filename().string().find("log_config") != std::string::npos) {
    //         log_config_paths.push_back(entry.path());
    //         std::cout << "found log config: " << entry.path() << std::endl;
    //     }
    // }
    
    auto [particle, frameNumber, sysPath, trajPath] = loadParticleFromRoot(root_path, trajectory_frame);

    std::filesystem::path integrator_config_path = get_path(sysPath, "integrator_config.json");
    ConfigDict integrator_config;
    integrator_config.from_json(integrator_config_path);

    
    return 0;
}
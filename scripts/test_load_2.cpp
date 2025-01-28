#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/particles/base/config.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

// TODO:
// create load functions for particles (connect with the non-derivable quantities in the particle)
// set packing fraction in particle configs
// be able to cleanly resume runs
// make functions for creating default configs for relevant particle types
// get old main branch, make a copy, move all files to copy, overwrite main with current code
// make python library for handling data (compress / decompress files, run simulations, etc)

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

    

    // IF RESUMING, DELETE THE ROWS IN THE ENERGY FILE THAT EXCEED THE CURRENT FRAME


    // // load positions from last trajectory step
    // // auto host_positions = read_array_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);
    // SwapData2D<double> positions = read_2d_swap_data_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);

    // // load radii and masses from system path
    // Data1D<double> radii = read_1d_data_from_file<double>(source_path + "system/init/radii.dat", particle_config.n_particles);
    // Data1D<double> masses = read_1d_data_from_file<double>(source_path + "system/init/masses.dat", particle_config.n_particles);

    // // try to load box size from last trajectory step - otherwise default to the system path
    // std::string box_size_path;
    // try {
    //     box_size_path = last_step_dir + "/box_size.dat";
    //     if (!std::filesystem::exists(box_size_path)) {
    //         box_size_path = source_path + "system/init/box_size.dat";
    //     }
    // } catch (std::filesystem::filesystem_error& e) {
    //     std::cerr << "Error loading box size: " << e.what() << std::endl;
    //     return 1;
    // }
    // auto box_size = read_array_from_file<double>(box_size_path, 2, 1);

    // // create the particle object
    // auto particle = create_particle(particle_config);

    // // set the data
    // particle->positions.setData(positions.getDataX(), positions.getDataY());
    // particle->radii.setData(radii.getData());
    // particle->masses.setData(masses.getData());
    // particle->setBoxSize(box_size);
    // particle->setupNeighbors(particle_config);  // maybe necessary if the initial data was from a compression - cell size shrinks with box size


    // double packing_fraction_increment = 1e-8;
    // double alpha = 1e-4;
    // double beta1 = 0.9;
    // double beta2 = 0.999;
    // double epsilon = 1e-8;
    // double avg_pe_target = 1e-16;
    // double avg_pe_diff_target = 1e-16;

    // AdamConfig adam_config(alpha, beta1, beta2, epsilon);
    // Adam adam(*particle, adam_config);

    // long num_adam_steps = 1e5;
    // long num_compression_steps = 1e6;
    // long num_energy_saves = 1e2;
    // long num_state_saves = 1e3;
    // long min_state_save_decade = 1e1;

    // // long save_every_N_steps = 1e3;
    // // std::vector<LogGroupConfig> log_group_configs = {
    // //     config_from_names_lin_everyN({"step", "PE/N", "phi"}, save_every_N_steps, "console"),  // logs to the console
    // //     config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    // //     config_from_names_lin_everyN({"step", "PE", "phi"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
    // //     config_from_names_lin_everyN({"positions", "forces", "box_size", "cell_index", "cell_start"}, save_every_N_steps, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    // // };
    // // IOManager jamming_io_manager(log_group_configs, *particle, &adam, target_path, 1, true);
    // // jamming_io_manager.write_params();  // TODO: move this into the io manager constructor
    
    // // // use adam and the regular jamming routine
    // // jam_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, packing_fraction_increment * 1e-4, 1.01);

    // // // use adam and compress/decompress to a target packing fraction
    // // compress_to_phi_adam(*particle, adam, jamming_io_manager, 1e5, num_adam_steps, avg_pe_target, avg_pe_diff_target, delta_phi);
    
    // // // use adam and compress until pe is above this target
    // // double max_pe_target = 1e-6;  // over-compress until pe is above here
    // // compress_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, max_pe_target);
    
    // // // use adam and decompress until pe is below this target
    // // double min_pe_target = 1e-16;  // decompress until pe is below here
    // // decompress_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, min_pe_target);

    // // run dynamics
    // particle->setRandomVelocities(dynamics_temperature);
    // NVEConfig nve_config(dt_dimless * particle->getTimeUnit());
    // NVE nve(*particle, nve_config);
    // long save_every_N_steps = 1e3;
    // std::vector<LogGroupConfig> log_group_configs = {
    //     config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
    //     config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    //     config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
    //     config_from_names_lin_everyN({"positions", "forces", "velocities", "force_pairs", "distance_pairs", "num_neighbors", "neighbor_list", "static_particle_index", "pair_ids"}, save_every_N_steps, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    // };
    // IOManager dynamics_io_manager(log_group_configs, *particle, &nve, target_path, 1, true);
    // dynamics_io_manager.write_params();
    // long step = 0;
    // while (step < num_dynamics_steps) {
    //     nve.step();
    //     dynamics_io_manager.log(step);
    //     step++;
    // }

    return 0;
}
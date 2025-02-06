#include "../include/constants.h"
#include "../include/particles/base/particle.h"
#include "../include/particles/disk/disk.h"
#include "../include/integrator/nve.h"
#include "../include/integrator/adam.h"
#include "../include/integrator/grad_desc.h"
#include "../include/io/orchestrator.h"
#include "../include/particles/particle_factory.h"
#include "../include/io/utils.h"
#include "../include/io/console_log.h"
#include "../include/io/energy_log.h"
#include "../include/io/io_manager.h"
#include "../include/routines/compression.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

#include <nlohmann/json.hpp>

// run the over-compression/decompression to first zero-PE state below jamming to find phi_j_over

int main() {
    for (int i = 0; i < 10; i++) {
        double decompression_amount = 1e-7;
        double dynamics_temperature = 1e-8;
        double dt_dimless = 1e-2;
        long num_dynamics_steps = 1e7;

        std::string root = "/home/mmccraw/dev/data/24-11-08/compressions-new/" + std::to_string(i) + "/";
        // std::string source_path = root + "/fine-jamming/";  // load jamming
        // std::string source_path = root + "/jamming/";  // load jamming
        // std::string target_path = root + "/fine-jamming/";  // [1] resume jamming using smaller steps to 'ideally' find phi_j


        double delta_phi = 1e-5;  // 1e-4 [x], 1e-5 [x], 1e-6 [x], 0, -1e-6 [x], -1e-5 [x], -1e-4 [x]
        // 1e-2 [x], 2e-2 [x], 1e-3 [x]
        // for lower temperature, do 2e-2 [x] and 1e-5 [x]
        // std::string target_path = root + "/over-jamming-offsets/" + std::to_string(delta_phi) + "/";  // create fine-jamming
        std::string source_path = root + "/over-jamming-offsets/" + std::to_string(delta_phi) + "/";  // load fine-jamming
        std::string target_path = root + "/over-jamming-dynamics-lower-temperature/" + std::to_string(delta_phi) + "-" + std::to_string(dynamics_temperature) + "/";
        
        // [2] overcompress [1] and decompress to first zero-PE state below jamming to find phi_j_over
        // [3] get compress/decompress [1] and [2] packings with offsets \Delta \phi = +/- [0, 1e-6, 1e-5, 1e-4]
        // run dynamics on all cases in [3]

        // load particle config from system path
        std::string particle_config_path = source_path + "system/particle_config.json";
        std::ifstream particle_config_input = open_input_file(particle_config_path);
        nlohmann::json particle_config_json;
        particle_config_input >> particle_config_json;
        particle_config_input.close();
        auto particle_config = BidisperseDiskConfig::from_json(particle_config_json);

        // get the directory of the last trajectory step
        std::string file_prefix = "t";
        long last_step = get_largest_file_index(source_path + "trajectories/", file_prefix);
        std::string last_step_dir = source_path + "trajectories/" + file_prefix + std::to_string(last_step);

        // load positions from last trajectory step
        // auto host_positions = read_array_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);
        SwapData2D<double> positions = read_2d_swap_data_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);

        // load radii and masses from system path
        Data1D<double> radii = read_1d_data_from_file<double>(source_path + "system/init/radii.dat", particle_config.n_particles);
        Data1D<double> masses = read_1d_data_from_file<double>(source_path + "system/init/masses.dat", particle_config.n_particles);

        // try to load box size from last trajectory step - otherwise default to the system path
        std::string box_size_path;
        try {
            box_size_path = last_step_dir + "/box_size.dat";
            if (!std::filesystem::exists(box_size_path)) {
                box_size_path = source_path + "system/init/box_size.dat";
            }
        } catch (std::filesystem::filesystem_error& e) {
            std::cerr << "Error loading box size: " << e.what() << std::endl;
            return 1;
        }
        auto box_size = read_array_from_file<double>(box_size_path, 2, 1);

        // create the particle object
        auto particle = create_particle(particle_config);

        // set the data
        particle->positions.setData(positions.getDataX(), positions.getDataY());
        particle->radii.setData(radii.getData());
        particle->masses.setData(masses.getData());
        particle->setBoxSize(box_size);
        particle->setupNeighbors(particle_config);  // maybe necessary if the initial data was from a compression - cell size shrinks with box size


        double packing_fraction_increment = 1e-8;
        double alpha = 1e-4;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        double avg_pe_target = 1e-16;
        double avg_pe_diff_target = 1e-16;

        AdamConfig adam_config(alpha, beta1, beta2, epsilon);
        Adam adam(*particle, adam_config);

        long num_adam_steps = 1e5;
        long num_compression_steps = 1e6;
        long num_energy_saves = 1e2;
        long num_state_saves = 1e3;
        long min_state_save_decade = 1e1;

        // long save_every_N_steps = 1e3;
        // std::vector<LogGroupConfig> log_group_configs = {
        //     config_from_names_lin_everyN({"step", "PE/N", "phi"}, save_every_N_steps, "console"),  // logs to the console
        //     config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        //     config_from_names_lin_everyN({"step", "PE", "phi"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
        //     config_from_names_lin_everyN({"positions", "forces", "box_size", "cell_index", "cell_start"}, save_every_N_steps, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // };
        // IOManager jamming_io_manager(log_group_configs, *particle, &adam, target_path, 1, true);
        // jamming_io_manager.write_params();  // TODO: move this into the io manager constructor
        
        // // use adam and the regular jamming routine
        // jam_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, packing_fraction_increment * 1e-4, 1.01);

        // // use adam and compress/decompress to a target packing fraction
        // compress_to_phi_adam(*particle, adam, jamming_io_manager, 1e5, num_adam_steps, avg_pe_target, avg_pe_diff_target, delta_phi);
        
        // // use adam and compress until pe is above this target
        // double max_pe_target = 1e-6;  // over-compress until pe is above here
        // compress_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, max_pe_target);
        
        // // use adam and decompress until pe is below this target
        // double min_pe_target = 1e-16;  // decompress until pe is below here
        // decompress_adam(*particle, adam, jamming_io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, min_pe_target);

        // run dynamics
        particle->setRandomVelocities(dynamics_temperature);
        NVEConfig nve_config(dt_dimless * particle->getTimeUnit());
        NVE nve(*particle, nve_config);
        long save_every_N_steps = 1e3;
        std::vector<LogGroupConfig> log_group_configs = {
            config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
            config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
            config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
            config_from_names_lin_everyN({"positions", "forces", "velocities", "force_pairs", "distance_pairs", "num_neighbors", "neighbor_list", "static_particle_index", "pair_ids"}, save_every_N_steps, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        };
        IOManager dynamics_io_manager(log_group_configs, *particle, &nve, target_path, 1, true);
        dynamics_io_manager.write_params();
        long step = 0;
        while (step < num_dynamics_steps) {
            nve.step();
            dynamics_io_manager.log(step);
            step++;
        }

    }
    return 0;
}
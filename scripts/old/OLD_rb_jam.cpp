#include "../include/constants.h"
#include "../include/particles/base/particle.h"
#include "../include/routines/compression.h"
#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particles/particle_factory.h"
#include "../include/integrator/adam.h"
#include "../include/io/utils.h"
#include "../include/io/console_log.h"
#include "../include/io/energy_log.h"
#include "../include/io/io_manager.h"
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
#include "../include/routines/initialization.h"
#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

#include <nlohmann/json.hpp>

int main() {

    // make a config for the rb partcle
    // set the config large vertex numbers after creation
    long n_vertices_per_small_particle = 26;
    long n_vertices_per_large_particle = 0;  // not known yet
    long n_vertices = 0;  // not known yet
    long n_particles = 256;

    double particle_mass = 1.0;
    double e_c = 1.0;
    double n_c = 2.0;

    long segment_length_per_vertex_diameter = 1.0;

    double packing_fraction = 0.6;

    double size_ratio = 1.4;
    double count_ratio = 0.5;

    long particle_dim_block = 256;
    long vertex_dim_block = 256;

    double vertex_neighbor_cutoff_multiplier = 1.5;  // vertices within this multiple of the maximum vertex diameter will be considered neighbors
    double vertex_neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the vertex neighbor cutoff, the vertex neighbor list will be updated

    double neighbor_cutoff_multiplier = 1.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.2;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double num_particles_per_cell = 8.0;  // the desired number of particles per cell
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated

    long seed = -1;
    bool rotation = true;
    double vertex_radius = 0.5;  // arbitrary value
    BidisperseRigidBumpyConfig config(
        seed, 
        n_particles,
        particle_mass,
        e_c,
        n_c,
        packing_fraction,
        neighbor_cutoff_multiplier,
        neighbor_displacement_multiplier,
        num_particles_per_cell,
        cell_displacement_multiplier,
        "cell",
        particle_dim_block,
        n_vertices,
        vertex_dim_block,
        vertex_neighbor_cutoff_multiplier,
        vertex_neighbor_displacement_multiplier,
        segment_length_per_vertex_diameter,
        rotation,
        vertex_radius,
        size_ratio,
        count_ratio,
        n_vertices_per_small_particle,
        n_vertices_per_large_particle
    );

    // for (long i = 0; i < 10; i++) {
    for (long i = 0; i < 1; i++) {
        std::string root = "/home/mmccraw/dev/data/24-11-08/rb-compressions-new/" + std::to_string(i) + "/";
        std::string source_path = root + "/jamming/";
        std::string target_path = root + "/jamming/";

        RigidBumpy rb;
        rb.initializeFromConfig(config);
        
        double packing_fraction_increment = 1e-4;
        long num_compression_steps = 3e6;
        long num_adam_steps = 1e5;
        long num_state_saves = 1e3;
        long num_energy_saves = 1e3;
        long num_saves = 1e4;

        double alpha = 1e-4;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        double avg_pe_target = 1e-16;
        double avg_pe_diff_target = 1e-16;

        std::string particle_config_path = source_path + "system/particle_config.json";
        std::ifstream particle_config_input = open_input_file(particle_config_path);
        nlohmann::json particle_config_json;
        particle_config_input >> particle_config_json;
        particle_config_input.close();
        auto particle_config = BidisperseRigidBumpyConfig::from_json(particle_config_json);

        // get the directory of the last trajectory step
        std::string file_prefix = "t";
        long last_step = get_largest_file_index(source_path + "trajectories/", file_prefix);
        std::string last_step_dir = source_path + "trajectories/" + file_prefix + std::to_string(last_step);
        std::string init_dir = source_path + "init/";

        // load from last trajectory step
        // angles.dat  box_size.dat  forces.dat  positions.dat  vertex_forces.dat  vertex_positions.dat
        // load from init
        // angular_velocities.dat  forces.dat  moments_of_inertia.dat        positions.dat  velocities.dat     vertex_masses.dat          vertex_positions.dat
        // box_size.dat            masses.dat  num_vertices_in_particle.dat  radii.dat      vertex_forces.dat  vertex_particle_index.dat
        SwapData2D<double> positions = read_2d_swap_data_from_file<double>(last_step_dir + "/positions.dat", particle_config.n_particles, 2);
        SwapData2D<double> vertex_positions = read_2d_swap_data_from_file<double>(last_step_dir + "/vertex_positions.dat", particle_config.n_vertices, 2);
        SwapData1D<double> angles = read_1d_swap_data_from_file<double>(last_step_dir + "/angles.dat", particle_config.n_particles);

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

        // set the data
        rb.positions.setData(positions.getDataX(), positions.getDataY());
        rb.vertex_positions.setData(vertex_positions.getDataX(), vertex_positions.getDataY());
        rb.angles.setData(angles.getData());
        rb.setBoxSize(box_size);
        rb.setupNeighbors(particle_config);  // maybe necessary if the initial data was from a compression - cell size shrinks with box size
        


        // AdamConfig adam_config(alpha, beta1, beta2, epsilon);
        // Adam adam(rb, adam_config);

        // // Make the io manager
        // std::vector<LogGroupConfig> log_group_configs = {
        //     config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e2, "console"),  // logs to the console
        //     config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size", "vertex_positions", "vertex_forces", "vertex_masses", "angular_velocities", "moments_of_inertia", "num_vertices_in_particle", "vertex_particle_index"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        //     config_from_names_lin_everyN({"positions", "forces", "angles", "box_size", "vertex_positions", "vertex_forces"}, num_saves, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        //     config_from_names_lin_everyN({"step", "PE", "phi"}, num_saves, "energy"),  // saves the energy data to the energy file
        // };
        // IOManager io_manager(log_group_configs, rb, &adam, target_path, 1, true);
        // io_manager.write_params();

        // jam_adam(rb, adam, io_manager, num_compression_steps, num_adam_steps, avg_pe_target, avg_pe_diff_target, packing_fraction_increment, packing_fraction_increment * 1e-2, 1.0001);
    }
    return 0;
}
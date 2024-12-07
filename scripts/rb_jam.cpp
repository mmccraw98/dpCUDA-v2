#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/particle/rigid_bumpy.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particle/particle_factory.h"
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
    long n_particles = 1024;

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

    long seed = 0;
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

    RigidBumpy rb;
    rb.initializeFromConfig(config);
    rb.initAdamVariables();
    
    // TODO: use shared memory for all vertex data for a single particle stored on a single block
    // make disk into a base class for point-like particles
    // make rigid bumpy into a base class for vertex-based particles
    
    double packing_fraction_target = 0.8;
    double packing_fraction_increment = 1e-4;
    long num_compression_steps = 1e5;
    long num_adam_steps = 1e5;
    long num_state_saves = 1e3;
    long num_energy_saves = 1e3;

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;

    AdamConfig adam_config(alpha, beta1, beta2, epsilon);
    Adam adam(rb, adam_config);

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        // config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e2, "console"),  // logs to the console
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e4, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size", "vertex_positions", "vertex_forces", "vertex_masses", "angular_velocities", "moments_of_inertia", "num_vertices_in_particle"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // config_from_names_log({"positions", "velocities"}, num_steps, num_state_saves, min_state_save_decade, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // config_from_names_log({"positions", "velocities", "forces", "angular_velocities", "angles"}, num_steps, num_state_saves, min_state_save_decade, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin({"positions", "forces", "angles"}, num_compression_steps, num_state_saves, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_compression_steps, num_energy_saves, "energy"),  // saves the energy data to the energy file
    };
    std::cout << "creating io manager" << std::endl;
    IOManager io_manager(log_group_configs, rb, &adam, "/home/mmccraw/dev/data/24-12-06/rb-jam", 1, true);
    std::cout << "writing params" << std::endl;
    io_manager.write_params();

    std::cout << "stepping" << std::endl;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();



    rb.calculateParticleArea();
    packing_fraction = rb.getPackingFraction();
    std::cout << "initial packing fraction: " << packing_fraction << std::endl;
    // exit(0);

    long compression_step = 0;
    double avg_pe_past_jamming = 1e-9;  // marks being above jamming (might be too high)
    double avg_pe = 0.0;
    while (packing_fraction < packing_fraction_target && compression_step < num_compression_steps) {
        long adam_step = 0;
        double dof = static_cast<double>(rb.n_dof);
        double last_avg_pe = 0.0;
        double avg_pe_diff = 0.0;
        while (adam_step < num_adam_steps) {
            adam.minimize(adam_step);
            avg_pe = rb.totalPotentialEnergy() / dof / rb.e_c;
            avg_pe_diff = std::abs(avg_pe - last_avg_pe);
            last_avg_pe = avg_pe;
            if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
                break;
            }
            adam_step++;
        }
        if (adam_step == num_adam_steps) {
            std::cout << "Adam failed to converge" << std::endl;
            break;
        }
        io_manager.log(compression_step);
        rb.scaleToPackingFraction(packing_fraction + packing_fraction_increment);
        packing_fraction = rb.getPackingFraction();
        compression_step++;
    }

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration * 1e-3 << " seconds for " << rb.n_particles << " particles" << std::endl;


    return 0;
}

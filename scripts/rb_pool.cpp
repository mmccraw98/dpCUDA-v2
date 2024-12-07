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

#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

#include <nlohmann/json.hpp>

int main() {
    long n_vertices_per_small_particle = 26;
    long n_vertices_per_large_particle = 0;  // not known yet
    long n_vertices = 0;  // not known yet
    long n_particles = 16;

    double particle_mass = 1.0;
    double e_c = 1.0;
    double n_c = 2.0;
    long segment_length_per_vertex_diameter = 1.0;
    double packing_fraction = 0.6;
    double size_ratio = 1.0;
    double count_ratio = 0.5;
    long particle_dim_block = 256;
    long vertex_dim_block = 256;
    double vertex_neighbor_cutoff_multiplier = 1.5;
    double vertex_neighbor_displacement_multiplier = 0.5;
    double neighbor_cutoff_multiplier = 1.5;
    double neighbor_displacement_multiplier = 0.2;
    double num_particles_per_cell = 8.0;
    double cell_displacement_multiplier = 0.5;

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
        "verlet",
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

    bool random_angles = true;
    double cue_speed = 0.01;
    double cue_angle = 0.0;

    double length = 2.0;
    double width = 1.0;

    // make the pool balls
    double ball_radius = 0.04;
    double rack_x = 0.75 * length;
    double rack_y = 0.5 * width;
    double spacing = ball_radius * 2.0 * 1.1;

    SwapData2D<double> source_positions;
    SwapData2D<double> source_velocities;
    SwapData1D<double> source_radii;
    thrust::host_vector<double> host_positions_x(n_particles, 0.0);
    thrust::host_vector<double> host_positions_y(n_particles, 0.0);
    thrust::host_vector<double> host_velocities_x(n_particles, 0.0);
    thrust::host_vector<double> host_velocities_y(n_particles, 0.0);
    thrust::host_vector<double> host_radii(n_particles, ball_radius);
    host_positions_x[0] = 0.25 * length;
    host_positions_y[0] = rack_y;
    host_velocities_x[0] = cue_speed * std::cos(cue_angle);
    host_velocities_y[0] = cue_speed * std::sin(cue_angle);
    long ball_index = 1;
    for (long row = 0; row < 5; row++) {
        for (long col = 0; col < row + 1; col++) {
            double x = rack_x + row * spacing * std::cos(M_PI / 6.0);
            double y = rack_y + (col - row / 2.0) * spacing;
            host_positions_x[ball_index] = x;
            host_positions_y[ball_index] = y;
            ball_index++;
        }
    }
    source_positions.resizeAndFill(n_particles, 0.0, 0.0);
    source_velocities.resizeAndFill(n_particles, 0.0, 0.0);
    source_radii.resizeAndFill(n_particles, ball_radius);
    source_positions.setData(host_positions_x, host_positions_y);
    source_velocities.setData(host_velocities_x, host_velocities_y);

    rotation = config.rotation;
    rb.segment_length_per_vertex_diameter = config.segment_length_per_vertex_diameter;
    rb.initializeVerticesFromDiskPacking(source_positions, source_radii, config.n_vertex_per_small_particle, config.particle_dim_block, config.vertex_dim_block);
    rb.define_unique_dependencies();
    rb.setSeed(config.seed);

    if (random_angles) {
        rb.angles.fillRandomUniform(0, 2 * M_PI, 0, config.seed);
    }
    rb.velocities.copyFrom(source_velocities);

    thrust::host_vector<double> host_box_size(N_DIM);
    host_box_size[0] = length;
    host_box_size[1] = width;
    rb.setBoxSize(host_box_size);

    config.vertex_radius = rb.getVertexRadius();
    double geom_scale = rb.getGeometryScale();
    config.e_c *= (geom_scale * geom_scale);
    rb.setEnergyScale(config.e_c, "c");
    rb.setExponent(config.n_c, "c");
    rb.setMass(config.mass);
    rb.config = std::make_unique<BidisperseRigidBumpyConfig>(config);
    rb.setNeighborMethod(config.neighbor_list_update_method);
    rb.setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
    rb.setCellSize(config.num_particles_per_cell, config.cell_displacement_multiplier);
    rb.max_vertex_neighbors_allocated = 8;
    rb.syncVertexNeighborList();
    rb.initNeighborList();
    rb.syncVertexNeighborList();

    // make the integrator
    double dt_dimless = 1e-2;  // 1e-3 might be the best option
    NVEConfig nve_config(dt_dimless * rb.getTimeUnit() * rb.getGeometryScale());
    NVE nve(rb, nve_config);

    long num_steps = 1e6;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e3;
    long min_state_save_decade = 1e1;

    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size", "vertex_positions", "vertex_velocities"}, "init"),
        config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_steps, num_energy_saves, "energy"),
        config_from_names_lin({"positions", "velocities", "vertex_positions", "vertex_velocities", "forces"}, num_steps, num_state_saves, "state"),
    };

    IOManager io_manager(log_group_configs, rb, &nve, "/home/mmccraw/dev/data/24-12-07/rb-pool-1", 1, true);
    io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.wall_step();
        io_manager.log(step);
        step++;
    }
    return 0;
}

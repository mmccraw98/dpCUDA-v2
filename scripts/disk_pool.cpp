#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particle/particle_factory.h"
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

#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

#include <nlohmann/json.hpp>

int main() {
    double neighbor_cutoff_multiplier = 1.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.2;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double num_particles_per_cell = 8.0;  // the desired number of particles per cell
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated
    BidisperseDiskConfig config(0, 16, 1.0, 1e2, 2.0, 0.8, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, "verlet", 256, 1.0, 0.5);

    double cue_speed = 1.0;
    double cue_angle = 0.0;

    Disk disk;
    disk.initializeFromConfig(config);
    double length = 2.0;
    double width = 1.0;
    thrust::host_vector<double> host_box_size(N_DIM);
    host_box_size[0] = length;
    host_box_size[1] = width;
    disk.setBoxSize(host_box_size);

    // make the pool balls
    double ball_radius = 0.04;
    double rack_x = 0.75 * length;
    double rack_y = 0.5 * width;
    double spacing = ball_radius * 2.0 * 1.01;

    thrust::host_vector<double> host_positions_x = disk.positions.getDataX();
    thrust::host_vector<double> host_positions_y = disk.positions.getDataY();
    thrust::host_vector<double> host_velocities_x = disk.velocities.getDataX();
    thrust::host_vector<double> host_velocities_y = disk.velocities.getDataY();
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
    disk.positions.setData(host_positions_x, host_positions_y);
    disk.radii.fill(ball_radius);
    disk.velocities.setData(host_velocities_x, host_velocities_y);

    // make the integrator
    double dt_dimless = 1e-2;  // 1e-3 might be the best option
    NVEConfig nve_config(dt_dimless * disk.getTimeUnit());
    NVE nve(disk, nve_config);

    long num_steps = 1e6;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e3;
    long min_state_save_decade = 1e1;

    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),
        config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_steps, num_energy_saves, "energy"),
        config_from_names_lin({"positions", "velocities", "forces"}, num_steps, num_state_saves, "state"),
    };

    IOManager io_manager(log_group_configs, disk, &nve, "/home/mmccraw/dev/data/24-11-08/jamming/pool-1", 1, true);
    io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.wall_step();
        io_manager.log(step);
        step++;
    }
    return 0;
}

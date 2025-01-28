#include "../include/constants.h"
#include "../include/particles/base/particle.h"
#include "../include/particles/disk/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particles/particle_factory.h"
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
    BidisperseDiskConfig config(0, 1024, 1.0, 1.0, 2.0, 0.8, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, "cell", 256, 1.4, 0.5);
    auto particle = create_particle(config);
    particle->setRandomVelocities(1e-4);
    double dt_dimless = 1e-2;  // 1e-3 might be the best option
    NVEConfig nve_config(dt_dimless * particle->getTimeUnit());
    std::cout << "dt: " << nve_config.dt << std::endl;
    NVE nve(*particle, nve_config);
    long num_steps = 1e5;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e2;
    long min_state_save_decade = 1e1;
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size", "boxSize"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_steps, num_energy_saves, "energy"),  // saves the energy data to the energy file
        config_from_names_lin_everyN({"positions", "velocities", "forces", "cell_index", "particle_index", "static_particle_index", "cell_start", "num_neighbors", "neighbor_list", "kinetic_energy", "potential_energy", "particlePos", "boxSize"}, 100, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    };
    IOManager io_manager(log_group_configs, *particle, &nve, "/home/mmccraw/dev/data/24-11-08/jamming/1-run", 16, true);
    io_manager.write_params();  // TODO: move this into the io manager constructor
    long step = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (step < num_steps) {
        nve.step();
        io_manager.log(step);
        step++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration * 1e-3 << " seconds for " << particle->n_particles << " particles" << std::endl;
    return 0;
}

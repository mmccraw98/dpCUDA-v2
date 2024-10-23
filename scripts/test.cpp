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

#include "../include/data/data_1d.h"
#include "../include/data/data_2d.h"

#include <nlohmann/json.hpp>

int main() {

    // make a function that defines a system of disks and then minimizes their overlaps to a target potential energy using adam
    double neighbor_cutoff_multiplier = 1.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double num_particles_per_cell = 8.0;  // the desired number of particles per cell
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated
    BidisperseDiskConfig config(0, 1024 * 100, 1.0, 1.0, 2.0, 0.8, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, "cell", 256, 1.4, 0.5);
    auto particle = create_particle(config);

    particle->initAdamVariables();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;


    AdamConfig adam_config(alpha, beta1, beta2, epsilon);
    Adam adam(*particle, adam_config);

    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e3, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, *particle, &adam, "", 1, true);

    long step = 0;
    long num_steps = 1e5;
    while (step < num_steps) {
        adam.step(step);
        io_manager.log(step);
        step++;
    }


    // RigidBumpy rb;

    // long num_particles = 32;
    // long num_vertices_per_particle = 32;
    // rb.setParticleCounts(num_particles, num_vertices_per_particle * num_particles);

    // long particle_dim_block = 256;
    // long vertex_dim_block = 256;
    // rb.setKernelDimensions(particle_dim_block, vertex_dim_block);

    // rb.initDynamicVariables();


    return 0;
}

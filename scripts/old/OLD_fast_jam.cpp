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
    BidisperseDiskConfig config(-1, 512, 1.0, 1.0, 2.0, 0.6, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, "cell", 256, 2.4, 0.05);
    auto particle = create_particle(config);
    
    // make the integrator
    particle->initAdamVariables();

    double packing_fraction_increment = 0.0001;
    double packing_fraction_target = 0.85;
    double packing_fraction = particle->getPackingFraction();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;

    AdamConfig adam_config(alpha, beta1, beta2, epsilon);
    Adam adam(*particle, adam_config);

    long num_adam_steps = 1e5;
    long num_compression_steps = 1e4;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e3;
    long min_state_save_decade = 1e1;

    // Make the io manager
    long save_every_N_steps = 50;
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 100, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "boxSize", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin_everyN({"step", "PE", "phi"}, save_every_N_steps, "energy"),  // saves the energy data to the energy file
        config_from_names_lin_everyN({"particlePos", "forces", "boxSize", "cell_index", "particle_index", "static_particle_index", "cell_start", "num_neighbors", "neighbor_list"}, save_every_N_steps, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
    };

    IOManager io_manager(log_group_configs, *particle, &adam, "/home/mmccraw/dev/data/24-11-08/jamming/25", 1, true);
    io_manager.write_params();  // TODO: move this into the io manager constructor

    long compression_step = 0;
    while (packing_fraction < packing_fraction_target && compression_step < num_compression_steps) {
        long adam_step = 0;
        double dof = static_cast<double>(particle->n_dof);
        double last_avg_pe = 0.0;
        double avg_pe_diff = 0.0;
        while (adam_step < num_adam_steps) {
            adam.minimize(adam_step);
            double avg_pe = particle->totalPotentialEnergy() / dof;
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
        particle->scaleToPackingFraction(packing_fraction + packing_fraction_increment);
        packing_fraction = particle->getPackingFraction();
        compression_step++;
    }
    return 0;
}
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
    long n_particles = 3;

    double particle_mass = 1.0;
    double e_c = 1.0;
    double n_c = 2.0;

    long segment_length_per_vertex_diameter = 1.0;

    double packing_fraction = 0.5;

    double size_ratio = 1.4;
    double count_ratio = 0.5;

    long particle_dim_block = 256;
    long vertex_dim_block = 256;

    double vertex_neighbor_cutoff_multiplier = 1.5;  // vertices within this multiple of the maximum vertex diameter will be considered neighbors
    double vertex_neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the vertex neighbor cutoff, the vertex neighbor list will be updated

    double neighbor_cutoff_multiplier = 1.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double num_particles_per_cell = 8.0;  // the desired number of particles per cell
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated

    long seed = 0;

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
        size_ratio,
        count_ratio,
        n_vertices_per_small_particle,
        n_vertices_per_large_particle
    );

    RigidBumpy rb;
    rb.config = std::make_unique<BidisperseRigidBumpyConfig>(config);

    // TODO: use shared memory for all vertex data for a single particle stored on a single block

    // make disk into a base class for point-like particles

    // make rigid bumpy into a base class for vertex-based particles

    // make a function that defines a system of disks and then minimizes their overlaps to a target potential energy using adam
    BidisperseDiskConfig disk_config(config.seed, config.n_particles, config.mass, config.e_c, config.n_c, config.packing_fraction, config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier, config.num_particles_per_cell, config.cell_displacement_multiplier, config.neighbor_list_update_method, particle_dim_block, size_ratio, count_ratio);
    auto [positions, radii] = get_minimal_overlap_positions_and_radii(disk_config);
    double disk_area = 0.0;
    thrust::host_vector<double> h_radii = radii.getData();
    for (double radius : h_radii) {
        disk_area += PI * radius * radius;
    }



    rb.segment_length_per_vertex_diameter = config.segment_length_per_vertex_diameter;
    
    rb.initializeVerticesFromDiskPacking(positions, radii, config.n_vertex_per_small_particle, config.particle_dim_block, config.vertex_dim_block);

    rb.define_unique_dependencies();

    rb.setSeed(config.seed);

    // set random angles
    rb.angles.fillRandomUniform(0, 2 * M_PI, 0, config.seed);

    std::cout << "done with initializeVerticesFromDiskPacking" << std::endl;

    rb.calculateParticleArea();

    rb.initializeBox(config.packing_fraction);

    rb.setEnergyScale(config.e_c, "c");
    rb.setExponent(config.n_c, "c");
    rb.setMass(config.mass);

    double vertex_diameter = 2.0 * rb.getVertexRadius();
    double particle_diameter = rb.getDiameter("max");

    rb.vertex_particle_neighbor_cutoff = particle_diameter;  // particles within this distance of a vertex will be checked for vertex neighbors
    rb.vertex_neighbor_cutoff = 2.0 * vertex_diameter;  // vertices within this distance of each other are neighbors

    // rb.updateVerletList();

    rb.setNeighborMethod("verlet");
    rb.setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
    rb.max_vertex_neighbors_allocated = 4;

    // init the neighbor list for the particles    
    rb.initVerletList();


    double force_balance = rb.getForceBalance();
    std::cout << "force_balance: " << force_balance << std::endl;
    rb.calculateForces();
    force_balance = rb.getForceBalance();
    std::cout << "force_balance: " << force_balance << std::endl;
    // this->calculateForces();  // make sure forces are calculated before the integration starts
    // // may want to check that the forces are balanced




    double dt_dimless = 1e-2;  // 1e-3 might be the best option
    NVEConfig nve_config(dt_dimless * rb.getTimeUnit());
    std::cout << "dt: " << nve_config.dt << std::endl;
    NVE nve(rb, nve_config);

    long num_steps = 1e5;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e3;
    long min_state_save_decade = 1e1;
    std::cout << "num_steps: " << num_steps << std::endl;

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
        config_from_names({"radii", "masses", "positions", "velocities", "forces", "box_size"}, "init"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // config_from_names_log({"positions", "velocities"}, num_steps, num_state_saves, min_state_save_decade, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // config_from_names_lin({"positions", "velocities", "forces"}, num_steps, num_state_saves, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_steps, num_energy_saves, "energy"),  // saves the energy data to the energy file
    };
    std::cout << "creating io manager" << std::endl;
    IOManager io_manager(log_group_configs, rb, &nve, "/home/mmccraw/dev/data/24-10-14/working-on-bumpy/rb1", 4, true);
    std::cout << "writing params" << std::endl;
    io_manager.write_params();



    return 0;
}

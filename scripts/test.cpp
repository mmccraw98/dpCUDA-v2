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

    // TODO: use shared memory for all vertex data for a single particle stored on a single block

    // make disk into a base class for point-like particles

    // make rigid bumpy into a base class for vertex-based particles

    // make a function that defines a system of disks and then minimizes their overlaps to a target potential energy using adam
    double neighbor_cutoff_multiplier = 1.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double num_particles_per_cell = 8.0;  // the desired number of particles per cell
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated
    BidisperseDiskConfig config(0, 32, 1.0, 1.0, 2.0, 0.8, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, num_particles_per_cell, cell_displacement_multiplier, "cell", 256, 1.4, 0.5);
    auto [positions, radii] = get_minimal_overlap_positions_and_radii(config);

    RigidBumpy rb;


    long num_particles = 32;
    long num_vertices_per_particle = 32;
    rb.setParticleCounts(num_particles, num_vertices_per_particle * num_particles);

    long particle_dim_block = 256;
    long vertex_dim_block = 256;
    rb.setKernelDimensions(particle_dim_block, vertex_dim_block);

    rb.initDynamicVariables();
    rb.initGeometricVariables();
    
    // set the particle angles
 
    rb.define_unique_dependencies();

    rb.setSeed(config.seed);

    // // initialize the vertices on the circles
    long num_vertices_in_small_particle = 26;
    rb.initializeVerticesFromDiskPacking(positions, radii, num_vertices_in_small_particle);

    std::cout << "done with initializeVerticesFromDiskPacking" << std::endl;

    rb.initializeBox(config.packing_fraction);

    rb.setEnergyScale(config.e_c, "c");
    rb.setExponent(config.n_c, "c");
    rb.setMass(config.mass);

    

    // this->setNeighborMethod(config.neighbor_list_update_method);
    // this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);

    // if (this->neighbor_list_update_method == "cell") {
    //     bool could_set_cell_size = this->setCellSize(config.num_particles_per_cell, config.cell_displacement_multiplier);
    //     if (!could_set_cell_size) {
    //         std::cout << "WARNING: Disk::initializeFromConfig: Could not set cell size.  Attempting to use verlet list instead." << std::endl;
    //         this->setNeighborMethod("verlet");
    //     }
    //     bool could_set_neighbor_size = this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
    //     if (!could_set_neighbor_size) {
    //         std::cerr << "ERROR: Disk::initializeFromConfig: Could not set neighbor size for cell list - neighbor cutoff exceeds box size.  Attempting to use all-to-all instead." << std::endl;
    //         this->setNeighborMethod("all");
    //     }
    // }
    // if (this->neighbor_list_update_method == "verlet") {
    //     bool could_set_neighbor_size = this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
    //     if (!could_set_neighbor_size) {
    //         std::cout << "WARNING: Disk::initializeFromConfig: Could not set neighbor size.  Attempting to use all-to-all instead." << std::endl;
    //         this->setNeighborMethod("all");
    //     }
    // }
    // this->initNeighborList();
    // this->calculateForces();  // make sure forces are calculated before the integration starts
    // // may want to check that the forces are balanced

    return 0;
}

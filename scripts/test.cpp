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

    rb.segment_length_per_vertex_diameter = 1.0;


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

    num_particles = rb.positions.size[0];
    thrust::host_vector<double> hpx = rb.positions.x.getData();
    thrust::host_vector<double> hpy = rb.positions.y.getData();
    for (long i = 0; i < num_particles; i++) {
        std::cout << "particle " << i << " position: " << hpx[i] << ", " << hpy[i] << std::endl;
    }

    rb.calculateParticleArea();

    rb.initializeBox(config.packing_fraction);

    rb.setEnergyScale(config.e_c, "c");
    rb.setExponent(config.n_c, "c");
    rb.setMass(config.mass);

    double vertex_diameter = 2.0 * rb.getVertexRadius();
    double particle_diameter = rb.getDiameter("max");

    std::cout << "vertex_diameter: " << vertex_diameter << std::endl;
    std::cout << "particle_diameter: " << particle_diameter << std::endl;

    rb.vertex_particle_neighbor_cutoff = particle_diameter;  // particles within this distance of a vertex will be checked for vertex neighbors
    rb.vertex_neighbor_cutoff = 2.0 * vertex_diameter;  // vertices within this distance of each other are neighbors

    // rb.updateVerletList();

    rb.setNeighborMethod("verlet");
    rb.setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
    rb.max_vertex_neighbors_allocated = 4;

    // init the neighbor list for the particles    
    rb.initVerletList();


    // this->calculateForces();  // make sure forces are calculated before the integration starts
    // // may want to check that the forces are balanced

    return 0;
}

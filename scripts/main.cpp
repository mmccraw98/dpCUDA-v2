#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/io/logger.h"
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

int main() {

    // TODO: make a particle factory
    // TODO: make the file io (input works with particle factory) (may need to make a particle method to construct values that are missing (some values can be derived from others))


    // constructing the object

    Disk particle;

    particle.setSeed(0);

    // set/sync number of vertices/particles, define the array sizes
    particle.setParticleCounts(1024, 0);

    // set/sync kernel dimensions
    particle.setKernelDimensions(256);  // TODO: not sure how to best motivate this

    // define the particle sizes, initialize the box to a set packing fraction, and set random positions
    particle.setBiDispersity(1.4, 0.5);  // TODO: define scaling to determine geometry units (min, max, or mean)
    particle.initializeBox(0.5);
    particle.setRandomPositions();
    // define geometry when relevant (i.e. initialize vertex configurations, calculate shape parameters, etc.)

    // set/sync energies
    particle.setEnergyScale(1.0, "c");
    particle.setExponent(2.0, "c");
    particle.setMass(1.0);
    // TODO: set timestep
    
    particle.setRandomVelocities(1e-3);

    // define the neighbor cutoff size
    particle.setNeighborCutoff(1.5);  // 1.5 * min_diameter

    // update the neighbor list
    particle.updateNeighborList();


    // constructing the simulation:
    std::vector<std::string> log_entries = {"KE/N", "PE/N", "TE/N", "T", "TE"};
    Logger logger(particle, log_entries);

    logger.write_header();

    NVE nve(particle, 0.001);
    for (long i = 0; i < 1e4; i++) {
       nve.step();
       if (i % 1000 == 0) {
           logger.write_values(i);
       }
    }

    return 0;
}

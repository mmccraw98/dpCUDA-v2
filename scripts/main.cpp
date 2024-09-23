#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
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

int main() {

    // TODO: make a particle factory
    // TODO: make the file io (input works with particle factory) (may need to make a particle method to construct values that are missing (some values can be derived from others))

    // to make for the first time:
    // seed, particle counts, vertex counts, kernel dimensions, bidispersity values (2), packing fraction, energy scales, timestep, neighbor cutoff

    // to make from a file:
    // seed, particle counts, vertex counts, kernel dimensions, radii, masses, positions, velocities, energy values

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

    // make the integrator
    NVE nve(particle, 0.001);

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin({"step", "TE", "KE", "PE", "T"}, 1e4, 100, "energy"),
        config_from_names_lin_everyN({"step", "T", "TE/N"}, 1e2, "console")
    };


    IOManager io_manager(particle, nve, log_group_configs, "/home/mmccraw/dev/dpCUDA/old", false);

    io_manager.log(0);

    // TODO:
    // make state log
    // make state loading function (static method to load the particle from the file)
        // from file(file, )
    // make restart file and init file
    // make argument parser for defaults and overrides
    // add docstrings
    // may need to add an integrator get state method to allow integrator to save its variables

    // long step = 0;

    // while (step < 1e4) {
    //     nve.step();

    //     // WRAP THIS IN SOME FUNCTION:

    //     //////////////////////////////

    //     step++;
    // }

    return 0;
}

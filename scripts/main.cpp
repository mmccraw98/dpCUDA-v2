#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
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
    // constructing the object

    Disk disk;

    disk.setSeed(0);

    // set/sync number of vertices/particles, define the array sizes
    disk.setParticleCounts(1024*10, 0);

    // set/sync kernel dimensions
    disk.setKernelDimensions(256);  // TODO: not sure how to best motivate this

    // define the particle sizes, initialize the box to a set packing fraction, and set random positions
    disk.setBiDispersity(1.4, 0.5);  // TODO: define scaling to determine geometry units (min, max, or mean)
    disk.initializeBox(0.5);
    disk.setRandomPositions();
    // define geometry when relevant (i.e. initialize vertex configurations, calculate shape parameters, etc.)

    // set/sync energies
    disk.setEnergyScale(1.0, "c");
    disk.setExponent(2.0, "c");
    disk.setMass(1.0);
    // TODO: set timestep
    
    disk.setRandomVelocities(1e-3);

    // define the neighbor cutoff size
    disk.setNeighborCutoff(1.5);  // 1.5 * min_diameter

    // update the neighbor list
    disk.updateNeighborList();

    NVE nve(disk, 0.001);
    for (long i = 0; i < 1e5; i++) {
       nve.step();
       if (i % 1000 == 0) {
           disk.calculateKineticEnergy();
           std::cout << disk.totalEnergy() << std::endl;
       }
    }

    // constructing the simulation:

    return 0;
}

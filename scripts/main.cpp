#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
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
    Disk disk;

    disk.setSeed(0);
    disk.setEnergyScale(1.0, "c");
    disk.setExponent(2.0, "c");
    disk.setNumParticles(32);
    // disk.initDynamicVariables();
    // disk.initGeometricVariables();  // does nothing for disks
    disk.setKernelDimensions(256);  // not sure how to best motivate this
    // disk.setBiDispersity(1.4, 0.5);
    // disk.initializeBox();
    // disk.scaleToPackingFraction(0.5);
    // disk.setRandomPositions();

    // disk.updateNeighborList();
    // std::cout << "max_neighbors: " << disk.max_neighbors << std::endl;
    // std::cout << "max_neighbors_allocated: " << disk.max_neighbors_allocated << std::endl;

    // disk.updatePositions(0.01);
    return 0;
}

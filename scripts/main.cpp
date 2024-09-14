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
    std::cout << "Minimal CUDA Thrust and C++ Project" << std::endl;

    Disk disk(100, 0);

    disk.setBiDispersity(1.4, 0.5);
    disk.initializeBox(1.0);
    disk.scaleToPackingFraction(0.5);
    disk.setRandomPositions();

    std::cout << "Packing fraction: " << disk.getPackingFraction() << std::endl;

    disk.calculateKineticEnergy();

    std::cout << "Kinetic energy: " << disk.totalKineticEnergy() << std::endl;

    thrust::host_vector<double> h_kinetic_energy = disk.getArray<double>("d_kinetic_energy");

    std::cout << "Kinetic energy: " << h_kinetic_energy[0] << std::endl;

    return 0;
}

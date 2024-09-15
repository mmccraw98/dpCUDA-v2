#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
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

Disk::Disk() {
}

Disk::~Disk() {
}

// ----------------------------------------------------------------------
// ------------- Implementation of Pure Virtual Methods -----------------
// ----------------------------------------------------------------------

double Disk::getArea() const {
    return thrust::transform_reduce(d_radii.begin(), d_radii.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

double Disk::getOverlapFraction() const {
    return 0.0;
}

void Disk::calculateForces() {
    // Add logic to calculate forces between particles.
}

void Disk::calculateKineticEnergy() {
    std::cout << "FIXME: Implement calculateKineticEnergy" << std::endl;
    std::cout << "FIXME: Implement calculateKineticEnergy" << std::endl;
    std::cout << "FIXME: Implement calculateKineticEnergy" << std::endl;
    std::cout << "FIXME: Implement calculateKineticEnergy" << std::endl;
    std::cout << "FIXME: Implement calculateKineticEnergy" << std::endl;
    // thrust::transform(d_velocities.begin(), d_velocities.end(), d_kinetic_energy.begin(), Square());
}
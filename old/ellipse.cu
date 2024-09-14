#include "../../include/constants.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/ellipse.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/functors.h"
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

Ellipse::Ellipse() {
    std::cout << "Ellipse::Ellipse" << std::endl;
}

Ellipse::~Ellipse() {
    std::cout << "Ellipse::~Ellipse" << std::endl;
}

// ----------------------------------------------------------------------
// ------------- Implementation of Pure Virtual Methods -----------------
// ----------------------------------------------------------------------

void Ellipse::initDynamicVariables() {
    std::cout << "Ellipse::initDynamicVariables" << std::endl;
    Particle::initDynamicVariables();
    d_test_array.resize(n_particles);
}

void Ellipse::clearDynamicVariables() {
    std::cout << "Ellipse::clearDynamicVariables" << std::endl;
    Particle::clearDynamicVariables();
    d_test_array.clear();
}

void Ellipse::initGeometricVariables() {
    std::cout << "Ellipse::initGeometricVariables" << std::endl;
    // Add any necessary geometric variable initialization here if needed.
}

void Ellipse::clearGeometricVariables() {
    std::cout << "Ellipse::clearGeometricVariables" << std::endl;
    // Add any necessary geometric variable clearing here if needed.
}

double Ellipse::getArea() const {
    std::cout << "Ellipse::getArea" << std::endl;
    return thrust::transform_reduce(d_radii.begin(), d_radii.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

double Ellipse::getOverlapFraction() const {
    std::cout << "Ellipse::getOverlapFraction" << std::endl;
    std::cout << "TODO: Implement overlap fraction" << std::endl;
    return 0.0;
}

void Ellipse::scalePositions(double scale_factor) {
    std::cout << "Ellipse::scalePositions" << std::endl;
    thrust::transform(d_positions.begin(), d_positions.end(), thrust::make_constant_iterator(scale_factor), d_positions.begin(), thrust::multiplies<double>());
}

void Ellipse::updatePositions(double dt) {
    std::cout << "Ellipse::updatePositions" << std::endl;
    // Add logic to update positions based on time step.
}

void Ellipse::updateMomenta(double dt) {
    std::cout << "Ellipse::updateMomenta" << std::endl;
    // Add logic to update momenta based on time step.
}

void Ellipse::calculateForces() {
    std::cout << "Ellipse::calculateForces" << std::endl;
    // Add logic to calculate forces between particles.
}

void Ellipse::calculateKineticEnergy() {
    std::cout << "Ellipse::calculateKineticEnergy" << std::endl;
    thrust::transform(d_momenta.begin(), d_momenta.end(), d_kinetic_energy.begin(), Square());
}

void Ellipse::updateNeighborList() {
    std::cout << "Ellipse::updateNeighborList" << std::endl;
    // Add logic to update the neighbor list.
}

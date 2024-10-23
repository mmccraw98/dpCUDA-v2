#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/rigid_bumpy.h"
#include "../../include/kernels/kernels.cuh"
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

RigidBumpy::RigidBumpy() {
}

RigidBumpy::~RigidBumpy() {
}

// ----------------------------------------------------------------------
// --------------------- Overridden Methods -----------------------------
// ----------------------------------------------------------------------


void RigidBumpy::setKernelDimensions(long particle_dim_block, long vertex_dim_block) {
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "CUDA Info: Particle::setKernelDimensions: Max threads per block: " << maxThreadsPerBlock << std::endl;
    if (particle_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: Particle::setKernelDimensions: particle_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        particle_dim_block = maxThreadsPerBlock;
    }
    if (n_particles <= 0) {
        std::cout << "ERROR: Disk::setKernelDimensions: n_particles is 0.  Set n_particles before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_particles < particle_dim_block) {
        particle_dim_block = n_particles;
    }
    this->particle_dim_block = particle_dim_block;
    this->particle_dim_grid = (n_particles + particle_dim_block - 1) / particle_dim_block;

    if (vertex_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: RigidBumpy::setKernelDimensions: vertex_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        vertex_dim_block = maxThreadsPerBlock;
    }
    if (n_vertices <= 0) {
        std::cout << "ERROR: RigidBumpy::setKernelDimensions: n_vertices is 0.  Set n_vertices before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_vertices < vertex_dim_block) {
        vertex_dim_block = n_vertices;
    }
    this->vertex_dim_block = vertex_dim_block;
    this->vertex_dim_grid = (n_vertices + vertex_dim_block - 1) / vertex_dim_block;

    syncKernelDimensions();
}

void RigidBumpy::initDynamicVariables() {
    Particle::initDynamicVariables();
    vertex_positions.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_velocities.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_forces.resizeAndFill(n_vertices, 0.0, 0.0);
    angles.resizeAndFill(n_particles, 0.0);
    angular_velocities.resizeAndFill(n_particles, 0.0);
    torques.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::clearDynamicVariables() {
    Particle::clearDynamicVariables();
    vertex_positions.clear();
    vertex_velocities.clear();
    vertex_forces.clear();
    angles.clear();
    angular_velocities.clear();
    torques.clear();
}

double RigidBumpy::getArea() const {
    return 0.0;
}

double RigidBumpy::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void RigidBumpy::calculateForces() {
}

void RigidBumpy::calculateKineticEnergy() {
}
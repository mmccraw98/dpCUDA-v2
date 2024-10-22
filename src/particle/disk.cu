#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/disk.h"
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

Disk::Disk() {
}

Disk::~Disk() {
}

// ----------------------------------------------------------------------
// --------------------- Overridden Methods -----------------------------
// ----------------------------------------------------------------------


void Disk::setKernelDimensions(long particle_dim_block) {
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

    if (n_vertices > 0) {
        std::cout << "WARNING: Disk::setKernelDimensions: n_vertices is " << n_vertices << ".  This is being ignored." << std::endl;
    }

    syncKernelDimensions();
}

// ----------------------------------------------------------------------
// ------------- Implementation of Pure Virtual Methods -----------------
// ----------------------------------------------------------------------


double Disk::getArea() const {
    return thrust::transform_reduce(radii.d_vec.begin(), radii.d_vec.end(), Square(), 0.0, thrust::plus<double>()) * PI;
}

double Disk::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    // std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void Disk::calculateForces() {
    // kernelCalcDiskForces<<<particle_dim_grid, particle_dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_radii_ptr, d_forces_x_ptr, d_forces_y_ptr, d_potential_energy_ptr);
    kernelCalcDiskForces<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, forces.x.d_ptr, forces.y.d_ptr, potential_energy.d_ptr);
}

void Disk::calculateKineticEnergy() {
    // kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(d_velocities_x_ptr, d_velocities_y_ptr, d_masses_ptr, d_kinetic_energy_ptr);
    kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, kinetic_energy.d_ptr);
}

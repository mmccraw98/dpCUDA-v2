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

void RigidBumpy::initVertexVariables() {
    vertex_positions.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_velocities.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_forces.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_particle_index.resizeAndFill(n_vertices, 0);
    vertex_masses.resizeAndFill(n_vertices, 0.0);
}

void RigidBumpy::initDynamicVariables() {
    Particle::initDynamicVariables();
    angles.resizeAndFill(n_particles, 0.0);
    angular_velocities.resizeAndFill(n_particles, 0.0);
    torques.resizeAndFill(n_particles, 0.0);
    particle_start_index.resizeAndFill(n_particles, 0);
    num_vertices_in_particle.resizeAndFill(n_particles, 0);
}

void RigidBumpy::initGeometricVariables() {
    area.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::clearDynamicVariables() {
    Particle::clearDynamicVariables();
    vertex_positions.clear();
    vertex_velocities.clear();
    vertex_forces.clear();
    angles.clear();
    angular_velocities.clear();
    torques.clear();
    vertex_particle_index.clear();
    particle_start_index.clear();
    num_vertices_in_particle.clear();
}

void RigidBumpy::syncVertexIndices() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_particle_start_index_ptr, &particle_start_index.d_ptr, sizeof(particle_start_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_particle_start_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertices_in_particle_ptr, &num_vertices_in_particle.d_ptr, sizeof(num_vertices_in_particle.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_num_vertices_in_particle_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setParticleStartIndex() {
    thrust::exclusive_scan(num_vertices_in_particle.d_vec.begin(), num_vertices_in_particle.d_vec.end(), particle_start_index.d_vec.begin());
}

void RigidBumpy::syncVertexRadius(double vertex_radius) {
    cudaMemcpyToSymbol(d_vertex_radius, &vertex_radius, sizeof(double));
}

void RigidBumpy::initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle) {
    // i really dont like this

    // set the particle positions and radii from the disk packing
    positions.copyFrom(disk_positions);
    radii.copyFrom(disk_radii);
    
    double max_particle_diam = getDiameter("max");
    double min_particle_diam = getDiameter("min");



    auto num_small_particles = thrust::count_if(
        radii.d_vec.begin(), 
        radii.d_vec.end(), 
        [=] __device__ (double rad) {
            return rad == min_particle_diam / 2.0;
        }
    );


    auto num_large_particles = n_particles - num_small_particles;


    long num_vertices_in_large_particle = static_cast<long>(num_vertices_in_small_particle * max_particle_diam / min_particle_diam);

    double vertex_angle_small = 2 * M_PI / num_vertices_in_small_particle;
    double vertex_radius = min_particle_diam / (1 + segment_length_per_vertex_diameter / std::sin(vertex_angle_small / 2)) / 2.0;
    syncVertexRadius(vertex_radius);


    setNumVertices(num_small_particles * num_vertices_in_small_particle + num_large_particles * num_vertices_in_large_particle);

    // now need to re-update the vertex kernel dimensions
    setKernelDimensions(particle_dim_block, vertex_dim_block);

    initVertexVariables();


    
    kernelGetNumVerticesInParticles<<<particle_dim_grid, particle_dim_block>>>(
        radii.d_ptr, min_particle_diam, num_vertices_in_small_particle, max_particle_diam, num_vertices_in_large_particle, num_vertices_in_particle.d_ptr);



    setParticleStartIndex();
    kernelInitializeVerticesOnParticles<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, angles.d_ptr, vertex_particle_index.d_ptr, particle_start_index.d_ptr, num_vertices_in_particle.d_ptr, vertex_masses.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr);
    
    syncVertexIndices();
}

void RigidBumpy::calculateParticleArea() {
    kernelCalculateParticleArea<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, area.d_ptr);
}

double RigidBumpy::getParticleArea() const {
    return thrust::reduce(area.d_vec.begin(), area.d_vec.end(), 0.0, thrust::plus<double>());
}

void RigidBumpy::syncNeighborList() {
    Particle::syncNeighborList();
}
// void Particle::syncNeighborList() {
//     cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_neighbors_allocated, &this->max_neighbors_allocated, sizeof(this->max_neighbors_allocated));
//     if (cuda_err != cudaSuccess) {
//         std::cerr << "Particle::syncNeighborList: Error copying max_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
//         exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
//     }
//     cuda_err = cudaMemcpyToSymbol(d_neighbor_list_ptr, &neighbor_list.d_ptr, sizeof(neighbor_list.d_ptr));
//     if (cuda_err != cudaSuccess) {
//         std::cerr << "Particle::syncNeighborList: Error copying d_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
//         exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
//     }
//     cuda_err = cudaMemcpyToSymbol(d_num_neighbors_ptr, &num_neighbors.d_ptr, sizeof(num_neighbors.d_ptr));
//     if (cuda_err != cudaSuccess) {
//         std::cerr << "Particle::syncNeighborList: Error copying d_num_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
//         exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
//     }
// }

void RigidBumpy::setMass(double mass) {
    Particle::setMass(mass);
    vertex_masses.scale(mass);
    // check if sum of vertex masses is equal to particle mass for each particle
    double total_vertex_mass = thrust::reduce(vertex_masses.d_vec.begin(), vertex_masses.d_vec.end(), 0.0, thrust::plus<double>());
    if (std::abs(total_vertex_mass - mass * n_particles) > 1e-6) {
        std::cout << "WARNING: RigidBumpy::setMass: Total vertex mass does not match particle mass" << std::endl;
    }
}

double RigidBumpy::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void RigidBumpy::calculateForces() {
}

void RigidBumpy::calculateKineticEnergy() {
}
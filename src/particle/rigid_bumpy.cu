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
    cuda_err = cudaMemcpyToSymbol(d_vertex_particle_index_ptr, &vertex_particle_index.d_ptr, sizeof(vertex_particle_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_vertex_particle_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setParticleStartIndex() {
    thrust::exclusive_scan(num_vertices_in_particle.d_vec.begin(), num_vertices_in_particle.d_vec.end(), particle_start_index.d_vec.begin());
}

void RigidBumpy::syncVertexRadius(double vertex_radius) {
    cudaMemcpyToSymbol(d_vertex_radius, &vertex_radius, sizeof(double));
}

double RigidBumpy::getVertexRadius() {
    double vertex_radius;
    cudaMemcpyFromSymbol(&vertex_radius, d_vertex_radius, sizeof(double));
    return vertex_radius;
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
    std::cout << "vertex_angle_small: " << vertex_angle_small << std::endl;
    std::cout << "segment_length_per_vertex_diameter: " << segment_length_per_vertex_diameter << std::endl;
    std::cout << "sin(vertex_angle_small / 2): " << std::sin(vertex_angle_small / 2) << std::endl;
    std::cout << "min_particle_diam: " << min_particle_diam << std::endl;
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


// need to make a scale function for the particles which can then go into the base particle class and be overridden by the rigid bumpy class so we dont have to replicate the scaleToPackingFraction function

void RigidBumpy::calculateParticleArea() {
    std::cout << "RIGID BUMPY CALCULATE PARTICLE AREA" << std::endl;
    // kernelCalculateParticlePolygonArea<<<particle_dim_grid, particle_dim_block>>>(
    //     vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, area.d_ptr);
    kernelCalculateBumpyParticleAreaFull<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, radii.d_ptr, area.d_ptr);

    double min_diam = getDiameter("min");
    double max_diam = getDiameter("max");
    std::cout << "a: " << static_cast<double>(n_particles / 2) * M_PI * min_diam * min_diam / 4.0 + static_cast<double>(n_particles / 2) * M_PI * max_diam * max_diam / 4.0 << std::endl;
}

double RigidBumpy::getParticleArea() const {
    std::cout << "RIGID BUMPY GET PARTICLE AREA" << std::endl;
    double a = thrust::reduce(area.d_vec.begin(), area.d_vec.end(), 0.0, thrust::plus<double>());
    std::cout << "a: " << a << std::endl;
    return a;
}


// calculate the particle positions first if not already done not needed for rigid bumpy
// compare polygon areas to particle areas
// calculate the contribution to the area from the vertices
void RigidBumpy::scalePositions(double scale_factor) {
    kernelScalePositions<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, scale_factor
    );
}

void RigidBumpy::syncVertexNeighborList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_vertex_neighbors_allocated, &max_vertex_neighbors_allocated, sizeof(max_vertex_neighbors_allocated));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying max_vertex_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_vertex_neighbor_list_ptr, &vertex_neighbor_list.d_ptr, sizeof(vertex_neighbor_list.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_vertex_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertex_neighbors_ptr, &num_vertex_neighbors.d_ptr, sizeof(num_vertex_neighbors.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_num_vertex_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setMass(double mass) {
    Particle::setMass(mass);
    vertex_masses.scale(mass);
    // check if sum of vertex masses is equal to particle mass for each particle
    double total_vertex_mass = thrust::reduce(vertex_masses.d_vec.begin(), vertex_masses.d_vec.end(), 0.0, thrust::plus<double>());
    if (std::abs(total_vertex_mass / n_particles - mass) > 1e-6) {
        std::cout << "WARNING: RigidBumpy::setMass: Total vertex mass does not match particle mass" << std::endl;
    }
}

double RigidBumpy::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void RigidBumpy::calculateForces() {
}

void RigidBumpy::updatePositions() {
}

void RigidBumpy::updateVelocities() {
}

void RigidBumpy::calculateKineticEnergy() {
}

void RigidBumpy::calculateParticlePositions() {
    kernelCalculateParticlePositions<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr
    );
}

void RigidBumpy::updateVertexVerletList() {
    std::cout << "Updating vertex verlet list" << std::endl;
    vertex_neighbor_list.fill(-1L);
    kernelUpdateVertexNeighborList<<<vertex_dim_grid, vertex_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr, vertex_neighbor_cutoff, vertex_particle_neighbor_cutoff
    );
    long max_vertex_neighbors = thrust::reduce(num_vertex_neighbors.d_vec.begin(), num_vertex_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
    std::cout << "max_vertex_neighbors: " << max_vertex_neighbors << std::endl;
}

// void Particle::updateVerletList() {
//     neighbor_list.fill(-1L);
//     kernelUpdateNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neigh_displacements_sq.d_ptr, neighbor_cutoff);
//     max_neighbors = thrust::reduce(num_neighbors.d_vec.begin(), num_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
//     if (max_neighbors > max_neighbors_allocated) {
//         max_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_neighbors)));
//         std::cout << "Particle::updateVerletList: Resizing neighbor list to " << max_neighbors_allocated << std::endl;
//         neighbor_list.resize(n_particles * max_neighbors_allocated);
//         neighbor_list.fill(-1L);
//         syncNeighborList();
//         kernelUpdateNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neigh_displacements_sq.d_ptr, neighbor_cutoff);
//     }
// }

void RigidBumpy::initVerletListVariables() {
    Particle::initVerletListVariables();
    vertex_neighbor_list.resizeAndFill(n_vertices * max_vertex_neighbors_allocated, -1L);
    num_vertex_neighbors.resizeAndFill(n_vertices, 0L);
}

void RigidBumpy::initVerletList() {
    initVerletListVariables();
    syncNeighborList();
    syncVertexNeighborList();
    updateVerletList();
    updateVertexVerletList();
}
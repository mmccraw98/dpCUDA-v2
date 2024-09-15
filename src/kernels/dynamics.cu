#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/kernels/general.cuh"
#include "../../include/kernels/dynamics.cuh"

__global__ void kernelPrintN() {
    printf("d_n_particles: %ld\n", d_n_particles);
    printf("d_n_dim: %ld\n", d_n_dim);
    printf("d_n_vertices: %ld\n", d_n_vertices);
    printf("d_max_neighbors: %ld\n", d_max_neighbors);
    printf("d_max_neighbors_allocated: %ld\n", d_max_neighbors_allocated);
    printf("d_dim_block: %ld\n", d_dim_block);
    printf("d_dim_grid: %ld\n", d_dim_grid);
    printf("d_dim_vertex_grid: %ld\n", d_dim_vertex_grid);
    printf("d_box_size: %f\n", d_box_size[0]);
    printf("d_e_c: %f\n", d_e_c);
    printf("d_e_a: %f\n", d_e_a);
    printf("d_e_b: %f\n", d_e_b);
    printf("d_e_l: %f\n", d_e_l);
    printf("d_n_c: %f\n", d_n_c);
    printf("d_n_a: %f\n", d_n_a);
    printf("d_n_b: %f\n", d_n_b);
    printf("d_n_l: %f\n", d_n_l);
}

__global__ void kernelUpdatePositions(double* positions, const double* last_positions, double* displacements, double* velocities, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("particle_id: %ld d_n_particles: %ld\n", particle_id, d_n_particles);
    // Ensure you aren't exceeding the number of particles
    if (particle_id < 32) {
        printf("Processing particle_id: %ld\n", particle_id);
        
        #pragma unroll (N_DIM)
        for (long dim = 0; dim < d_n_dim; dim++) {
            positions[particle_id * d_n_dim + dim] += velocities[particle_id * d_n_dim + dim] * dt;
            displacements[particle_id * d_n_dim + dim] = positions[particle_id * d_n_dim + dim] - last_positions[particle_id * d_n_dim + dim];
        }
    }
}


__global__ void kernelUpdateVelocities(double* velocities, double* forces, const double* masses, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        #pragma unroll (N_DIM)
        for (long dim = 0; dim < d_n_dim; dim++) {
            velocities[particle_id * d_n_dim + dim] += forces[particle_id * d_n_dim + dim] / masses[particle_id] * dt;
        }
    }
}
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

__global__ void kernelUpdatePositions(double* positions, double* velocities, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        #pragma unroll (N_DIM)
        for (long dim = 0; dim < d_n_dim; dim++) {
            positions[particle_id * d_n_dim + dim] += velocities[particle_id * d_n_dim + dim] * dt;
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
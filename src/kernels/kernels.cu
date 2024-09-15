#include <stdio.h>
#include <cmath>
#include "../../include/constants.h"
#include "../../include/kernels/kernels.cuh"

// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------

__global__ void kernelUpdatePositions(double* positions, const double* last_positions, double* displacements, double* velocities, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("particle_id: %ld d_n_particles: %ld d_n_dim: %ld d_box_size: %f %f d_e_c: %f d_e_a: %f d_e_b: %f d_e_l: %f\n", particle_id, d_n_particles, d_n_dim, d_box_size[0], d_box_size[1], d_e_c, d_e_a, d_e_b, d_e_l);
    if (particle_id < d_n_particles) {
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


// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------

__global__ void kernelUpdateNeighborList(const double* positions, const double cutoff) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        printf("Particle %ld\n", particle_id);
        long added_neighbors = 0;
        double this_pos[N_DIM], other_pos[N_DIM];
        getPosition(particle_id, positions, this_pos);
        for (long other_id = 0; other_id < d_n_particles; other_id++) {
            if (particle_id != other_id) {
                getPosition(other_id, positions, other_pos);
                double distance = calcDistancePBC(this_pos, other_pos);
                if (distance < cutoff) {
                    d_neighbor_list_ptr[particle_id * d_max_neighbors + added_neighbors] = other_id;
                    added_neighbors++;
                }
            }
        }
        d_num_neighbors_ptr[particle_id] = added_neighbors;
    }
}
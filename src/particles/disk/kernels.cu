#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"
#include "../../../include/particles/base/kernels.cuh"
#include "../../../include/particles/disk/kernels.cuh"

// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalcDiskForces(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, double* __restrict__ forces_x, 
    double* __restrict__ forces_y, double* __restrict__ potential_energy) 
{

    // TODO: could probably make this faster by copying the relevant data into shared memory

    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];
    double force_acc_x = 0.0, force_acc_y = 0.0;
    double energy = 0.0;

    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;

        // Load neighbor's data
        double other_x = positions_x[other_id];
        double other_y = positions_y[other_id];
        double other_rad = radii[other_id];

        // Calculate force and energy using the interaction function
        double force_x, force_y;
        double interaction_energy = calcPointPointInteraction(
            pos_x, pos_y, rad, other_x, other_y, other_rad, 
            force_x, force_y
        );

        // Accumulate force and energy
        force_acc_x += force_x;
        force_acc_y += force_y;
        energy += interaction_energy;
    }

    // Store results in global memory
    forces_x[particle_id] += force_acc_x;
    forces_y[particle_id] += force_acc_y;
    potential_energy[particle_id] += energy;
}

__global__ void kernelCalcDiskWallForces(const double* positions_x, const double* positions_y, const double* radii, double* forces_x, double* forces_y, double* potential_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];
    double force_x = 0.0, force_y = 0.0;
    double interaction_energy = calcWallInteraction(pos_x, pos_y, rad, force_x, force_y);
    forces_x[particle_id] += force_x;
    forces_y[particle_id] += force_y;
    potential_energy[particle_id] += interaction_energy;
}
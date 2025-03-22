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

__global__ void kernelCalcDiskForceDistancePairs(const double* positions_x, const double* positions_y, double* potential_pairs, double* force_pairs_x, double* force_pairs_y, double* distance_pairs_x, double* distance_pairs_y, long* this_pair_id, long* other_pair_id, double* overlap_pairs, double* radsum_pairs, const double* radii, const long* static_particle_index, double* pair_separation_angle, double* hessian_pairs_x_x, double* hessian_pairs_x_y, double* hessian_pairs_y_x, double* hessian_pairs_y_y) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    long static_particle_id = static_particle_index[particle_id];
    if (particle_id >= d_n_particles) return;

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];
    double force_acc_x = 0.0, force_acc_y = 0.0;
    double energy = 0.0;

    for (long n = 0; n < d_max_neighbors_allocated; n++) {
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

        long pair_id = particle_id * d_max_neighbors_allocated + n;
        potential_pairs[pair_id] = interaction_energy;
        force_pairs_x[pair_id] = force_x;
        force_pairs_y[pair_id] = force_y;
        double x_dist = pbcDistance(pos_x, other_x, 0);
        double y_dist = pbcDistance(pos_y, other_y, 1);
        distance_pairs_x[pair_id] = x_dist;
        distance_pairs_y[pair_id] = y_dist;
        
        this_pair_id[pair_id] = static_particle_id;
        other_pair_id[pair_id] = static_particle_index[other_id];
        
        // this_pair_id[pair_id] = particle_id;
        // other_pair_id[pair_id] = other_id;

        
        double dist = sqrt(x_dist * x_dist + y_dist * y_dist);
        overlap_pairs[pair_id] = dist - (rad + other_rad);
        radsum_pairs[pair_id] = rad + other_rad;
        pair_separation_angle[pair_id] = atan2(y_dist, x_dist);

        if (dist < rad + other_rad) {
            double radsum = rad + other_rad;
            double tension = - d_e_c / radsum * std::pow(1 - dist / radsum, d_n_c - 1);
            double stiffness = d_e_c * (d_n_c - 1) / (radsum * radsum) * std::pow(1 - dist / radsum, d_n_c - 2);
            double x_dist_hat = x_dist / dist;
            double y_dist_hat = y_dist / dist;
            double hess_x_x = -stiffness * x_dist_hat * x_dist_hat - tension / dist * (1 - x_dist_hat * x_dist_hat);
            double hess_x_y = -stiffness * x_dist_hat * y_dist_hat + tension / dist * x_dist_hat * y_dist_hat;
            double hess_y_x = -stiffness * y_dist_hat * x_dist_hat + tension / dist * y_dist_hat * x_dist_hat;
            double hess_y_y = -stiffness * y_dist_hat * y_dist_hat - tension / dist * (1 - y_dist_hat * y_dist_hat);

            hessian_pairs_x_x[pair_id] = hess_x_x;
            hessian_pairs_x_y[pair_id] = hess_x_y;
            hessian_pairs_y_x[pair_id] = hess_y_x;
            hessian_pairs_y_y[pair_id] = hess_y_y;
        }
    }
}

__global__ void kernelCountDiskContacts(const double* positions_x, const double* positions_y, const double* radii, long* contact_counts) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];

    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;

        double other_x = positions_x[other_id];
        double other_y = positions_y[other_id];
        double other_rad = radii[other_id];

        double dist = sqrt(pbcDistance(pos_x, other_x, 0) * pbcDistance(pos_x, other_x, 0) + pbcDistance(pos_y, other_y, 1) * pbcDistance(pos_y, other_y, 1));
        if (dist < rad + other_rad) {
            contact_counts[particle_id]++;
        }
    }
}

__global__ void kernelCalcDiskStressTensor(const double* positions_x, const double* positions_y, const double* velocities_x, const double* velocities_y, const double* masses, const double* radii, double* stress_tensor_x_x, double* stress_tensor_x_y, double* stress_tensor_y_x, double* stress_tensor_y_y) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double mass = masses[particle_id];
    double rad = radii[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];

    // initialize stress tensor using the kinetic part
    double stress_tensor_x_x_acc = mass * vel_x * vel_x;
    double stress_tensor_x_y_acc = mass * vel_x * vel_y;
    double stress_tensor_y_x_acc = stress_tensor_x_y_acc;
    double stress_tensor_y_y_acc = mass * vel_y * vel_y;

    double box_area = d_box_size[0] * d_box_size[1];

    // calculate the virial part
    for (long n = 0; n < d_max_neighbors_allocated; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;

        double other_x = positions_x[other_id];
        double other_y = positions_y[other_id];
        double other_rad = radii[other_id];

        double x_dist = pbcDistance(pos_x, other_x, 0);
        double y_dist = pbcDistance(pos_y, other_y, 1);
        double dist = sqrt(x_dist * x_dist + y_dist * y_dist);
        
        double force_x, force_y;
        double interaction_energy = calcPointPointInteraction(
            pos_x, pos_y, rad, other_x, other_y, other_rad, 
            force_x, force_y
        );

        // divide by 2 to avoid double counting
        stress_tensor_x_x_acc += x_dist * force_x / 2;
        stress_tensor_x_y_acc += x_dist * force_y / 2;
        stress_tensor_y_x_acc += y_dist * force_x / 2;
        stress_tensor_y_y_acc += y_dist * force_y / 2;
    }

    stress_tensor_x_x[particle_id] = stress_tensor_x_x_acc / box_area;
    stress_tensor_x_y[particle_id] = stress_tensor_x_y_acc / box_area;
    stress_tensor_y_x[particle_id] = stress_tensor_y_x_acc / box_area;
    stress_tensor_y_y[particle_id] = stress_tensor_y_y_acc / box_area;
}
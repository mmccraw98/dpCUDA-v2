#include <stdio.h>
#include <cmath>
#include "../../include/constants.h"
#include "../../include/kernels/kernels.cuh"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

__constant__ long d_particle_dim_block;
__constant__ long d_particle_dim_grid;
__constant__ long d_vertex_dim_grid;
__constant__ long d_vertex_dim_block;
__constant__ long d_cell_dim_grid;
__constant__ long d_cell_dim_block;

__constant__ double d_vertex_radius;

__constant__ double d_box_size[N_DIM];

__constant__ long d_n_dim = N_DIM;
__constant__ long d_n_particles;
__constant__ long d_n_vertices;

__constant__ double d_e_c;
__constant__ double d_e_a;
__constant__ double d_e_b;
__constant__ double d_e_l;

__constant__ double d_n_c;
__constant__ double d_n_a;
__constant__ double d_n_b;
__constant__ double d_n_l;

__constant__ long* d_num_neighbors_ptr;
__constant__ long* d_neighbor_list_ptr;
__constant__ long d_max_neighbors_allocated;

__constant__ long* d_num_vertex_neighbors_ptr;
__constant__ long* d_vertex_neighbor_list_ptr;
__constant__ long d_max_vertex_neighbors_allocated;

__constant__ long* d_particle_start_index_ptr;
__constant__ long* d_num_vertices_in_particle_ptr;
__constant__ long* d_vertex_particle_index_ptr;

__constant__ long d_n_cells;
__constant__ long d_n_cells_dim;
__constant__ double d_cell_size;

// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------

// Should have the displacement calculation here rather than in velocities since this is only called once per time step
// can sneak in a lot of extra single-particle calculations here since it is called once per time step
__global__ void kernelUpdatePositions(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    const double* __restrict__ last_neigh_positions_x, const double* __restrict__ last_neigh_positions_y,
    const double* __restrict__ last_cell_positions_x, const double* __restrict__ last_cell_positions_y,
    double* __restrict__ neigh_displacements_sq, double* __restrict__ cell_displacements_sq,
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y, const double dt) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Load particle data into registers to minimize repeated global memory access
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];

    // Update positions
    pos_x += vel_x * dt;
    pos_y += vel_y * dt;
    positions_x[particle_id] = pos_x;
    positions_y[particle_id] = pos_y;

    // Calculate squared displacement for neighbor list
    double dx_neigh = pos_x - last_neigh_positions_x[particle_id];
    double dy_neigh = pos_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;
    // TODO: check for max displacement threshold here to avoid the thrust reduce call

    // Calculate squared displacement for cell list
    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;
    // TODO: check for max displacement threshold here to avoid the thrust reduce call
}


__global__ void kernelUpdateRigidPositions(double* positions_x, double* positions_y, double* angles, double* delta_x, double* delta_y, double* angle_delta, const double* last_neigh_positions_x, const double* last_neigh_positions_y, const double* last_cell_positions_x, const double* last_cell_positions_y, double* neigh_displacements_sq, double* cell_displacements_sq, const double* velocities_x, const double* velocities_y, const double* angular_velocities, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double angle = angles[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double ang_vel = angular_velocities[particle_id];

    // Update positions
    double temp_delta_x = vel_x * dt;
    double temp_delta_y = vel_y * dt;
    double temp_angle_delta = ang_vel * dt;
    delta_x[particle_id] = temp_delta_x;
    delta_y[particle_id] = temp_delta_y;
    angle_delta[particle_id] = temp_angle_delta;

    pos_x += temp_delta_x;
    pos_y += temp_delta_y;
    angle += temp_angle_delta;
    positions_x[particle_id] = pos_x;
    positions_y[particle_id] = pos_y;
    angles[particle_id] = angle;

    // Calculate squared displacement for neighbor list
    double dx_neigh = pos_x - last_neigh_positions_x[particle_id];
    double dy_neigh = pos_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;
    // TODO: check for max displacement threshold here to avoid the thrust reduce call

    // Calculate squared displacement for cell list
    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;
    // TODO: check for max displacement threshold here to avoid the thrust reduce call
}

// TODO: use the __restrict__ keyword for the pointers since they are not overlapping
// TODO: this could be parallelized over particles and dimensions
// This is also called 2x more than the position update so it would be good to keep it lightweight
__global__ void kernelUpdateVelocities(
    double* __restrict__ velocities_x, double* __restrict__ velocities_y,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    const double* __restrict__ masses, const double dt) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Load values into registers to minimize global memory access
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];
    double dt_over_mass = dt / masses[particle_id];

    // Update velocities
    vel_x += force_x * dt_over_mass;
    vel_y += force_y * dt_over_mass;

    // Store the results back to global memory
    velocities_x[particle_id] = vel_x;
    velocities_y[particle_id] = vel_y;
}

__global__ void kernelUpdateRigidVelocities(double* velocities_x, double* velocities_y, double* angular_velocities, const double* forces_x, const double* forces_y, const double* torques, const double* masses, const double* moments_of_inertia, const double dt, bool rotation) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Load values into registers to minimize global memory access
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];
    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double torque = torques[particle_id];
    double moment_of_inertia = moments_of_inertia[particle_id];
    double mass = masses[particle_id];
    double ang_vel = angular_velocities[particle_id];

    // Update velocities
    if (rotation) {
        ang_vel += torque * dt / moment_of_inertia;
    } else {
        ang_vel = 0.0;
    }
    vel_x += force_x * dt / mass;
    vel_y += force_y * dt / mass;

    // Store the results back to global memory
    velocities_x[particle_id] = vel_x;
    velocities_y[particle_id] = vel_y;
    angular_velocities[particle_id] = ang_vel;
}

__global__ void kernelTranslateAndRotateVertices1(const double* positions_x, const double* positions_y, double* vertex_positions_x, double* vertex_positions_y, const double* delta_x, const double* delta_y, const double* angle_delta) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;
    
    // Load particle data into registers to minimize repeated global memory access
    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double vertex_pos_x = vertex_positions_x[vertex_id];
    double vertex_pos_y = vertex_positions_y[vertex_id];
    double delta_x_vertex = delta_x[particle_id];
    double delta_y_vertex = delta_y[particle_id];
    double angle_delta_vertex = angle_delta[particle_id];

    // Get vertex position relative to particle center
    double rel_x = vertex_pos_x - pos_x;
    double rel_y = vertex_pos_y - pos_y;

    // Rotate
    double rot_x = rel_x * cos(angle_delta_vertex) - rel_y * sin(angle_delta_vertex);
    double rot_y = rel_x * sin(angle_delta_vertex) + rel_y * cos(angle_delta_vertex);

    // Translate
    vertex_positions_x[vertex_id] = rot_x + pos_x + delta_x_vertex;
    vertex_positions_y[vertex_id] = rot_y + pos_y + delta_y_vertex;
}



__global__ void kernelTranslateAndRotateVertices2(const double* positions_x, const double* positions_y, double* vertex_positions_x, double* vertex_positions_y, const double* delta_x, const double* delta_y, const double* angle_delta) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double delta_x_particle = delta_x[particle_id];
    double delta_y_particle = delta_y[particle_id];
    double angle_delta_particle = angle_delta[particle_id];
    double cos_angle = cos(angle_delta_particle);
    double sin_angle = sin(angle_delta_particle);

    // Apply the transformation to all vertices of the particle
    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        double vertex_pos_x = vertex_positions_x[vertex_id];
        double vertex_pos_y = vertex_positions_y[vertex_id];

        double rel_x = vertex_pos_x - pos_x;
        double rel_y = vertex_pos_y - pos_y;

        // Rotate
        double rot_x = rel_x * cos_angle - rel_y * sin_angle;
        double rot_y = rel_x * sin_angle + rel_y * cos_angle;

        // Translate
        vertex_positions_x[vertex_id] = rot_x + pos_x + delta_x_particle;
        vertex_positions_y[vertex_id] = rot_y + pos_y + delta_y_particle;
    }
}

__global__ void kernelRemoveMeanVelocities(double* __restrict__ velocities_x, double* __restrict__ velocities_y, const double mean_vel_x, const double mean_vel_y) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    velocities_x[particle_id] -= mean_vel_x;
    velocities_y[particle_id] -= mean_vel_y;
}


__global__ void kernelZeroForceAndPotentialEnergy(
    double* __restrict__ forces_x, 
    double* __restrict__ forces_y, 
    double* __restrict__ potential_energy
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    forces_x[particle_id] = 0.0;
    forces_y[particle_id] = 0.0;
    potential_energy[particle_id] = 0.0;
}


__global__ void kernelCalculateTranslationalKineticEnergy(
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ masses, double* __restrict__ kinetic_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Load data into registers to minimize global memory access
    double vx = velocities_x[particle_id];
    double vy = velocities_y[particle_id];
    double mass = masses[particle_id];

    // Compute squared velocity
    double velocity_sq = vx * vx + vy * vy;

    // Store kinetic energy
    kinetic_energy[particle_id] = 0.5 * mass * velocity_sq;
}


__global__ void kernelCalculateTranslationalAndRotationalKineticEnergy(
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ masses, const double* __restrict__ angular_velocities,
    const double* __restrict__ moments_of_inertia, double* __restrict__ kinetic_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double vel_x = velocities_x[particle_id];
    double vel_y = velocities_y[particle_id];
    double mass = masses[particle_id];
    double ang_vel = angular_velocities[particle_id];
    double moment_of_inertia = moments_of_inertia[particle_id];

    kinetic_energy[particle_id] = 0.5 * mass * (vel_x * vel_x + vel_y * vel_y) + 0.5 * moment_of_inertia * ang_vel * ang_vel;
}

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
    forces_x[particle_id] = force_acc_x;
    forces_y[particle_id] = force_acc_y;
    potential_energy[particle_id] = energy;
}

__global__ void kernelCalcRigidBumpyForces1(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, 
    double* __restrict__ vertex_forces_x, double* __restrict__ vertex_forces_y, double* __restrict__ vertex_torques, double* __restrict__ vertex_potential_energy) 
{
    // TODO: could probably make this faster by copying the relevant data into shared memory

    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    long num_neighbors = d_num_vertex_neighbors_ptr[vertex_id];
    // printf("vertex_id: %ld, num_neighbors: %ld\n", vertex_id, num_neighbors);
    if (num_neighbors == 0) return;

    double vertex_pos_x = vertex_positions_x[vertex_id];
    double vertex_pos_y = vertex_positions_y[vertex_id];
    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    double particle_pos_x = positions_x[particle_id];
    double particle_pos_y = positions_y[particle_id];
    double force_acc_x = 0.0, force_acc_y = 0.0;
    double energy = 0.0;

    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n];
        if (other_id == -1 || other_id == vertex_id) continue;

        // Load neighbor's data
        double other_vertex_pos_x = vertex_positions_x[other_id];
        double other_vertex_pos_y = vertex_positions_y[other_id];

        // Calculate force and energy using the interaction function
        double force_x, force_y;
        double interaction_energy = calcPointPointInteraction(
            vertex_pos_x, vertex_pos_y, d_vertex_radius, other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, 
            force_x, force_y
        );

        // Accumulate force and energy
        force_acc_x += force_x;
        force_acc_y += force_y;
        energy += interaction_energy;
    }

    // Calculate torque
    double torque = calcTorque(force_acc_x, force_acc_y, vertex_pos_x, vertex_pos_y, particle_pos_x, particle_pos_y);

    // Store results in global memory
    vertex_forces_x[vertex_id] = force_acc_x;
    vertex_forces_y[vertex_id] = force_acc_y;
    vertex_torques[vertex_id] = torque;
    vertex_potential_energy[vertex_id] = energy;
}

__global__ void kernelCalcRigidBumpyParticleForces1(
    const double* __restrict__ vertex_forces_x, const double* __restrict__ vertex_forces_y, const double* __restrict__ vertex_torques, const double* __restrict__ vertex_potential_energy,
    double* __restrict__ particle_forces_x, double* __restrict__ particle_forces_y, double* __restrict__ particle_torques, double* __restrict__ particle_potential_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double force_acc_x = 0.0, force_acc_y = 0.0;
    double torque_acc = 0.0;
    double energy = 0.0;

    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        force_acc_x += vertex_forces_x[vertex_id];
        force_acc_y += vertex_forces_y[vertex_id];
        torque_acc += vertex_torques[vertex_id];
        energy += vertex_potential_energy[vertex_id];
    }

    particle_forces_x[particle_id] = force_acc_x;
    particle_forces_y[particle_id] = force_acc_y;
    particle_torques[particle_id] = torque_acc;
    particle_potential_energy[particle_id] = energy;
}

__global__ void kernelCalcRigidBumpyForces2(const double* __restrict__ positions_x, const double* __restrict__ positions_y, const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, double* __restrict__ particle_forces_x, double* __restrict__ particle_forces_y, double* __restrict__ particle_torques, double* __restrict__ particle_potential_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double force_acc_x = 0.0, force_acc_y = 0.0;
    double torque_acc = 0.0;
    double energy = 0.0;
    double particle_pos_x = positions_x[particle_id];
    double particle_pos_y = positions_y[particle_id];
    double other_vertex_pos_x, other_vertex_pos_y;

    // loop over all vertices of the particle
    for (long v = 0; v < d_num_vertices_in_particle_ptr[particle_id]; v++) {
        long vertex_id = d_particle_start_index_ptr[particle_id] + v;
        double vertex_pos_x = vertex_positions_x[vertex_id];
        double vertex_pos_y = vertex_positions_y[vertex_id];
        long num_neighbors = d_num_vertex_neighbors_ptr[vertex_id];
        for (long n = 0; n < num_neighbors; n++) {
            long other_id = d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + n];
            if (other_id == -1 || other_id == vertex_id) continue;
            other_vertex_pos_x = vertex_positions_x[other_id];
            other_vertex_pos_y = vertex_positions_y[other_id];
            double temp_force_x, temp_force_y;
            double interaction_energy = calcPointPointInteraction(vertex_pos_x, vertex_pos_y, d_vertex_radius, other_vertex_pos_x, other_vertex_pos_y, d_vertex_radius, temp_force_x, temp_force_y);
            force_acc_x += temp_force_x;
            force_acc_y += temp_force_y;
            torque_acc += calcTorque(temp_force_x, temp_force_y, vertex_pos_x, vertex_pos_y, particle_pos_x, particle_pos_y);
            energy += interaction_energy;
        }
    }

    particle_forces_x[particle_id] = force_acc_x;
    particle_forces_y[particle_id] = force_acc_y;
    particle_torques[particle_id] = torque_acc;
    particle_potential_energy[particle_id] = energy;
}

// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------

__global__ void kernelUpdateNeighborList(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y, 
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    double* __restrict__ neigh_displacements_sq,
    const double cutoff) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long added_neighbors = 0;
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double cutoff_sq = cutoff * cutoff;

    // Iterate over all other particles
    for (long other_id = 0; other_id < d_n_particles; other_id++) {
        if (particle_id == other_id) continue; // Skip self

        double other_x = positions_x[other_id];
        double other_y = positions_y[other_id];

        // if (particle_id == 0) {
        //     printf("particle_id: %ld, particle_pos: %f, %f, other_id: %ld, other_pos: %f, %f, cutoff_sq: %f\n", particle_id, pos_x, pos_y, other_id, other_x, other_y, cutoff_sq);
        // }

        // Check if within cutoff using early exit
        if (isWithinCutoffSquared(pos_x, pos_y, other_x, other_y, cutoff_sq)) {
            if (added_neighbors < d_max_neighbors_allocated) {
                d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
            }
            added_neighbors++;
        }
    }
    last_neigh_positions_x[particle_id] = pos_x;
    last_neigh_positions_y[particle_id] = pos_y;
    neigh_displacements_sq[particle_id] = 0.0;
    d_num_neighbors_ptr[particle_id] = added_neighbors;
}


__global__ void kernelGetCellIndexForParticle(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y, 
    long* __restrict__ cell_index, long* __restrict__ particle_index) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;  // Early exit to avoid unnecessary nesting

    // Load positions into registers to minimize global memory access
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];

    // Compute cell indices with PBC wrapping
    long x_index = getPBCCellIndex(pos_x);
    long y_index = getPBCCellIndex(pos_y);

    // Compute linear cell index
    long linear_cell_id = x_index + y_index * d_n_cells_dim;

    // Store results directly to global memory
    cell_index[particle_id] = linear_cell_id;
    particle_index[particle_id] = particle_id;  // reassign the particle index to the particle id, but not the static particle index
}

__global__ void kernelGetFirstParticleIndexForCell(const long* __restrict__ cell_index, long* __restrict__ cell_start, const long width_offset, const long width) {
    long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= d_n_cells) return;

    bool found = false;
    for (long i = 0; i < d_n_particles; i++) {
        // printf("Cell index: %ld, cell id: %ld\n", cell_index[i], i);
        if (cell_index[i] == cell_id) {
            cell_start[cell_id] = i;
            found = true;
            break;
        }
    }
    if (!found) {
        cell_start[cell_id] = -1;
    }



    // if (cell_id == 0 && cell_index[0] == cell_id) {
    //     cell_start[cell_id] = 0;
    //     return;
    // }

    // // expand left
    // long left = max(0L, (cell_id - width_offset) * width);
    // while (left > 0 && cell_index[left] > cell_id) {
    //     left = max(0L, left - width);
    // }

    // // expand right
    // long right = min(d_n_particles - 1, (cell_id + width_offset - 1) * width);
    // while (right < d_n_particles - 1 && cell_index[right] < cell_id) {
    //     right = min(d_n_particles - 1, right + width);
    // }

    // // binary search to find an occurance of i
    // long mid;
    // bool found = false;
    // while (left <= right) {
    //     mid = (left + right) / 2;
    //     if (cell_index[mid] == cell_id) {
    //         found = true;
    //         break;
    //     } else if (cell_index[mid] < cell_id) {
    //         left = mid + 1;
    //     } else {
    //         right = mid - 1;
    //     }
    // }

    // if (found) {
    //     // while (cell_index[left] == cell_id && left > 0) {
    //     //     left--;
    //     // }
    //     // cell_start[cell_id] = left;
    //     // find the leftmost occurance of cell_id using another binary search
    //     right = mid;
    //     while (cell_index[left] == cell_id) {
    //         left -= width;
    //     }

    //     while (left < right) {
    //         mid = (left + right) / 2;
    //         if (cell_index[mid] == cell_id) {
    //             right = mid;
    //         } else {
    //             left = mid + 1;
    //         }
    //     }
    //     cell_start[cell_id] = left;
    // }

    // other option
// __global__ void kernelGetFirstParticleIndexForCell(
//     const long* __restrict__ sorted_cell_index, 
//     long* __restrict__ cell_start, 
//     const long width_offset, 
//     const long width) 
// {
    // long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (cell_id >= d_n_cells) return;

    // // Default: No particles found for this cell
    // cell_start[cell_id] = -1;

    // // Calculate search bounds with clamping
    // long left = max(0L, (cell_id - width_offset) * width);
    // long right = min(d_n_particles - 1, (cell_id + width_offset - 1) * width);

    // // Binary search to find the leftmost occurrence of cell_id
    // while (left < right) {
    //     long mid = (left + right) / 2;
    //     long mid_value = cell_index[mid];  // Load into register

    //     if (mid_value < cell_id) {
    //         left = mid + 1;
    //     } else {
    //         right = mid;
    //     }
    // }

    // // Verify if we found the cell_id at the left index
    // if (cell_index[left] == cell_id) {
    //     cell_start[cell_id] = left;
    // }
}

__global__ void kernelUpdateCellNeighborList(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    const double cutoff, const long* __restrict__ cell_index,
    const long* __restrict__ cell_start, double* __restrict__ neigh_displacements_sq) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_id >= d_n_particles) return;  // Avoid unnecessary nesting

    long added_neighbors = 0;
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double cutoff_sq = cutoff * cutoff;

    long cell_id = cell_index[particle_id];
    long cell_x = cell_id % d_n_cells_dim;
    long cell_y = cell_id / d_n_cells_dim;

    // Iterate over neighboring cells (unroll small loops)
    #pragma unroll
    for (long cell_y_offset = -1; cell_y_offset <= 1; cell_y_offset++) {
        long y_index = mod(cell_y + cell_y_offset, d_n_cells_dim);

        #pragma unroll
        for (long cell_x_offset = -1; cell_x_offset <= 1; cell_x_offset++) {
            long x_index = mod(cell_x + cell_x_offset, d_n_cells_dim);
            long neighbor_cell_id = x_index + y_index * d_n_cells_dim;

            // Get the particle range for this neighbor cell
            long start_id = cell_start[neighbor_cell_id];
            if (start_id == -1) continue;  // No particles in this cell

            long end_id = cell_start[neighbor_cell_id + 1];
            while (end_id == -1) {  // this is a hack to deal with the fact that some cells are empty
                neighbor_cell_id++;
                end_id = cell_start[neighbor_cell_id + 1];
            }


            // Loop over particles in the neighboring cell
            for (long neighbor_id = start_id; neighbor_id < end_id; neighbor_id++) {

                if (particle_id == neighbor_id) continue;  // Skip self

                // Load neighbor particle positions directly
                double other_x = positions_x[neighbor_id];
                double other_y = positions_y[neighbor_id];

                if (isWithinCutoffSquared(pos_x, pos_y, other_x, other_y, cutoff_sq)) {
                    if (added_neighbors < d_max_neighbors_allocated) {
                        d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = neighbor_id;
                    }
                    added_neighbors++;
                }
            }
        }
    }

    // Update the number of neighbors for this particle and reset the neighbor displacements
    d_num_neighbors_ptr[particle_id] = added_neighbors;
    neigh_displacements_sq[particle_id] = 0.0;
    last_neigh_positions_x[particle_id] = pos_x;
    last_neigh_positions_y[particle_id] = pos_y;
    // printf("particle_id: %ld, added_neighbors: %ld\n", particle_id, added_neighbors);
}

// TODO: add __restrict__ back
__global__ void kernelReorderParticleData(
	const long* __restrict__ particle_index,
	const double* __restrict__ positions_x, const double* __restrict__ positions_y,
	const double* __restrict__ forces_x, const double* __restrict__ forces_y,
	const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
	const double* __restrict__ masses, const double* __restrict__ radii,
	double* __restrict__ temp_positions_x, double* __restrict__ temp_positions_y,
	double* __restrict__ temp_forces_x, double* __restrict__ temp_forces_y,
	double* __restrict__ temp_velocities_x, double* __restrict__ temp_velocities_y,
	double* __restrict__ temp_masses, double* __restrict__ temp_radii,
	double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
	double* __restrict__ cell_displacements_sq) {

    // This is the new index of the particle in the sorted list
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_id >= d_n_particles) return;

    // This is the old index of the particle in the unsorted list
    long old_particle_id = particle_index[particle_id];

    // Copy the data from the old index to the new index
    temp_positions_x[particle_id] = positions_x[old_particle_id];
    temp_positions_y[particle_id] = positions_y[old_particle_id];
    temp_forces_x[particle_id] = forces_x[old_particle_id];
    temp_forces_y[particle_id] = forces_y[old_particle_id];
    temp_velocities_x[particle_id] = velocities_x[old_particle_id];
    temp_velocities_y[particle_id] = velocities_y[old_particle_id];
    temp_masses[particle_id] = masses[old_particle_id];
    temp_radii[particle_id] = radii[old_particle_id];

    // Reset the last cell positions and cell displacements since the cell list has been rebuilt
    last_cell_positions_x[particle_id] = positions_x[old_particle_id];
    last_cell_positions_y[particle_id] = positions_y[old_particle_id];
    cell_displacements_sq[particle_id] = 0.0;
}

__global__ void kernelReorderRigidBumpyParticleData(
    const long* __restrict__ particle_index,
    long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ num_vertices_in_particle,
    long* __restrict__ num_vertices_in_particle_new,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ positions_x_new, double* __restrict__ positions_y_new,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double* __restrict__ forces_x_new, double* __restrict__ forces_y_new,
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    double* __restrict__ velocities_x_new, double* __restrict__ velocities_y_new,
    const double* __restrict__ angular_velocities, const double* __restrict__ torques,
    double* __restrict__ angular_velocities_new, double* __restrict__ torques_new,
    const double* __restrict__ masses, const double* __restrict__ radii,
    double* __restrict__ masses_new, double* __restrict__ radii_new,
    const double* __restrict__ moments_of_inertia,
    double* __restrict__ moments_of_inertia_new,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    double* __restrict__ cell_displacements_sq) {

    long new_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_particle_id >= d_n_particles) return;

    long old_particle_id = particle_index[new_particle_id];
    
    // Create mapping from old particle indices to new particle indices
    old_to_new_particle_index[old_particle_id] = new_particle_id;

    // Reorder particle data
    positions_x_new[new_particle_id] = positions_x[old_particle_id];
    positions_y_new[new_particle_id] = positions_y[old_particle_id];
    forces_x_new[new_particle_id] = forces_x[old_particle_id];
    forces_y_new[new_particle_id] = forces_y[old_particle_id];
    velocities_x_new[new_particle_id] = velocities_x[old_particle_id];
    velocities_y_new[new_particle_id] = velocities_y[old_particle_id];
    angular_velocities_new[new_particle_id] = angular_velocities[old_particle_id];
    torques_new[new_particle_id] = torques[old_particle_id];
    masses_new[new_particle_id] = masses[old_particle_id];
    radii_new[new_particle_id] = radii[old_particle_id];
    moments_of_inertia_new[new_particle_id] = moments_of_inertia[old_particle_id];
    num_vertices_in_particle_new[new_particle_id] = num_vertices_in_particle[old_particle_id];
    last_cell_positions_x[new_particle_id] = positions_x[old_particle_id];
    last_cell_positions_y[new_particle_id] = positions_y[old_particle_id];
    cell_displacements_sq[new_particle_id] = 0.0;
}

__global__ void kernelReorderRigidBumpyVertexData(
    const long* __restrict__ vertex_particle_index,
    long* __restrict__ vertex_particle_index_new,
    const long* __restrict__ old_to_new_particle_index,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ particle_start_index_new,
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ vertex_positions_x_new, double* __restrict__ vertex_positions_y_new,
    const long* __restrict__ static_vertex_index,
    long* __restrict__ static_vertex_index_new) {

    long old_vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (old_vertex_id >= d_n_vertices) return;

    long old_particle_id = vertex_particle_index[old_vertex_id];
    long new_particle_id = old_to_new_particle_index[old_particle_id];
    long relative_vertex_id = old_vertex_id - particle_start_index[old_particle_id];
    long new_vertex_id = particle_start_index_new[new_particle_id] + relative_vertex_id;
    vertex_particle_index_new[new_vertex_id] = new_particle_id;
    vertex_positions_x_new[new_vertex_id] = vertex_positions_x[old_vertex_id];
    vertex_positions_y_new[new_vertex_id] = vertex_positions_y[old_vertex_id];
    static_vertex_index_new[new_vertex_id] = static_vertex_index[old_vertex_id];
}

// ----------------------------------------------------------------------
// --------------------- Minimizers -------------------------------
// ----------------------------------------------------------------------

__global__ void kernelAdamStep(
    double* __restrict__ first_moment_x, double* __restrict__ first_moment_y,
    double* __restrict__ second_moment_x, double* __restrict__ second_moment_y,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double alpha, double beta1, double beta2, double one_minus_beta1_pow_t, 
    double one_minus_beta2_pow_t, double epsilon, double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    double* __restrict__ neigh_displacements_sq, double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    double* __restrict__ cell_displacements_sq) {

    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    // Prefetch forces into registers
    double force_x = forces_x[particle_id];
    double force_y = forces_y[particle_id];

    // Load moments into registers
    double first_m_x = first_moment_x[particle_id];
    double first_m_y = first_moment_y[particle_id];
    double second_m_x = second_moment_x[particle_id];
    double second_m_y = second_moment_y[particle_id];

    // Update moments using fma for better performance
    first_m_x = fma(beta1, first_m_x, (beta1 - 1) * force_x);
    first_m_y = fma(beta1, first_m_y, (beta1 - 1) * force_y);

    second_m_x = fma(beta2, second_m_x, (1 - beta2) * force_x * force_x);
    second_m_y = fma(beta2, second_m_y, (1 - beta2) * force_y * force_y);

    // Compute bias-corrected moments
    double m_hat_x = first_m_x / one_minus_beta1_pow_t;
    double m_hat_y = first_m_y / one_minus_beta1_pow_t;

    double v_hat_x = second_m_x / one_minus_beta2_pow_t;
    double v_hat_y = second_m_y / one_minus_beta2_pow_t;

    // Compute position updates
    double update_x = -alpha * m_hat_x / (sqrt(v_hat_x) + epsilon);
    double update_y = -alpha * m_hat_y / (sqrt(v_hat_y) + epsilon);

    // Update positions
    positions_x[particle_id] += update_x;
    positions_y[particle_id] += update_y;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];

    double dx_neigh = pos_x - last_neigh_positions_x[particle_id];
    double dy_neigh = pos_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;

    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;

    // Store updated moments back
    first_moment_x[particle_id] = first_m_x;
    first_moment_y[particle_id] = first_m_y;
    second_moment_x[particle_id] = second_m_x;
    second_moment_y[particle_id] = second_m_y;
}

__global__ void kernelGradDescStep(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ forces_x, double* __restrict__ forces_y,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y, double* __restrict__ neigh_displacements_sq,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y, double* __restrict__ cell_displacements_sq,
    double alpha) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    positions_x[particle_id] += alpha * forces_x[particle_id];
    positions_y[particle_id] += alpha * forces_y[particle_id];

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];

    // Calculate squared displacement for neighbor list
    double dx_neigh = pos_x - last_neigh_positions_x[particle_id];
    double dy_neigh = pos_y - last_neigh_positions_y[particle_id];
    neigh_displacements_sq[particle_id] = dx_neigh * dx_neigh + dy_neigh * dy_neigh;

    // Calculate squared displacement for cell list
    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;
}


// ----------------------------------------------------------------------

__global__ void kernelGetNumVerticesInParticles(
    const double* __restrict__ radii,
    const double min_particle_diam,
    const long num_vertices_in_small_particle,
    const double max_particle_diam,
    const long num_vertices_in_large_particle,
    long* __restrict__ num_vertices_in_particle) {
    
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double radius = radii[particle_id];
    if (radius == min_particle_diam / 2.0) {
        num_vertices_in_particle[particle_id] = num_vertices_in_small_particle;
    } else if (radius == max_particle_diam / 2.0) {
        num_vertices_in_particle[particle_id] = num_vertices_in_large_particle;
    } else {
        printf("Error: particle radius %f is not equal to min or max particle diameter\n", radius);
    }
}

__global__ void kernelInitializeVerticesOnParticles(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, const double* __restrict__ angles,
    long* __restrict__ vertex_particle_index,
    const long* __restrict__ particle_start_index,
    const long* __restrict__ num_vertices_in_particle,
    double* __restrict__ vertex_masses,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y) {
    
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double angle = angles[particle_id];
    double radius = radii[particle_id];
    long num_vertices = num_vertices_in_particle[particle_id];
    long vertex_start_index = particle_start_index[particle_id];
    double vertex_angle_increment = 2 * M_PI / num_vertices;
    double inner_radius = radius - d_vertex_radius;


    for (long i = 0; i < num_vertices; i++) {
        double vertex_angle = i * vertex_angle_increment + angle;
        double vertex_x = positions_x[particle_id] + inner_radius * cos(vertex_angle);
        double vertex_y = positions_y[particle_id] + inner_radius * sin(vertex_angle);
        vertex_positions_x[vertex_start_index + i] = vertex_x;
        vertex_positions_y[vertex_start_index + i] = vertex_y;
        vertex_particle_index[vertex_start_index + i] = particle_id;
        vertex_masses[vertex_start_index + i] = 1 / static_cast<double>(num_vertices);
    }
}

__global__ void kernelGetVertexMasses(const double* __restrict__ radii, double* __restrict__ vertex_masses, const double* __restrict__ particle_masses) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;
    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    vertex_masses[vertex_id] = particle_masses[particle_id] / static_cast<double>(d_num_vertices_in_particle_ptr[particle_id]);
}

__global__ void kernelGetMomentsOfInertia(const double* __restrict__ positions_x, const double* __restrict__ positions_y, const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y, const double* __restrict__ vertex_masses, double* __restrict__ moments_of_inertia) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];
    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double mass = vertex_masses[first_vertex_index];  // assuming uniform mass for vertices within a particle
    double rel_pos_x = 0.0;
    double rel_pos_y = 0.0;
    double moment_of_inertia = 0.0;
    for (long i = 0; i < num_vertices; i++) {
        rel_pos_x = vertex_positions_x[first_vertex_index + i] - pos_x;
        rel_pos_y = vertex_positions_y[first_vertex_index + i] - pos_y;
        moment_of_inertia += mass * (rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y);
    }
    moments_of_inertia[particle_id] = moment_of_inertia;
}

// area

__global__ void kernelCalculateParticlePolygonArea(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_area
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double temp_area = 0.0;
    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Prefetch first vertex to avoid duplicate reads
    double pos_x = vertex_positions_x[first_vertex_index];
    double pos_y = vertex_positions_y[first_vertex_index];

    // Loop over vertices to compute the area
    for (long i = 1; i <= num_vertices; i++) {
        long next_index = first_vertex_index + (i % num_vertices);
        double next_pos_x = vertex_positions_x[next_index];
        double next_pos_y = vertex_positions_y[next_index];

        temp_area += pos_x * next_pos_y - next_pos_x * pos_y;

        // Move to the next vertex
        pos_x = next_pos_x;
        pos_y = next_pos_y;
    }
    particle_area[particle_id] = abs(temp_area) * 0.5;
}

__global__ void kernelCalculateBumpyParticleAreaFull(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ vertex_radii,
    double* __restrict__ particle_area
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double temp_polygon_area = 0.0;
    double exposed_vertex_area = 0.0;
    double temp_exposed_vertex_area = 0.0;
    double vertex_area = M_PI * d_vertex_radius * d_vertex_radius;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Load initial positions
    double pos_x = vertex_positions_x[first_vertex_index];
    double pos_y = vertex_positions_y[first_vertex_index];
    double prev_pos_x = vertex_positions_x[first_vertex_index + num_vertices - 1];
    double prev_pos_y = vertex_positions_y[first_vertex_index + num_vertices - 1];



    // Loop over vertices to compute the area
    for (long i = 0; i < num_vertices; i++) {
        temp_exposed_vertex_area = 0.0;

        // Calculate next vertex index with wrapping
        long next_index = first_vertex_index + ((i + 1) % num_vertices);
        double next_pos_x = vertex_positions_x[next_index];
        double next_pos_y = vertex_positions_y[next_index];

        // Calculate polygon area
        temp_polygon_area += pos_x * next_pos_y - next_pos_x * pos_y;

        // Calculate angle area
        double angle = angleBetweenVectors(next_pos_x, next_pos_y, pos_x, pos_y, prev_pos_x, prev_pos_y);

        // Calculate next vertex overlap area
        double dx = next_pos_x - pos_x;
        double dy = next_pos_y - pos_y;
        double r_ij = sqrt(dx * dx + dy * dy);

        if (r_ij < 2 * d_vertex_radius - 1e-10) {  // have to give some offset to prevent numerical errors - not good!
            temp_exposed_vertex_area -= calcOverlapLenseArea(r_ij, d_vertex_radius, d_vertex_radius) / 2.0;
        }

        // calculate previous vertex overlap area
        double prev_dx = prev_pos_x - pos_x;
        double prev_dy = prev_pos_y - pos_y;
        double prev_r_ij = sqrt(prev_dx * prev_dx + prev_dy * prev_dy);

        if (prev_r_ij < 2 * d_vertex_radius - 1e-10) {  // have to give some offset to prevent numerical errors - not good!
            temp_exposed_vertex_area -= calcOverlapLenseArea(prev_r_ij, d_vertex_radius, d_vertex_radius) / 2.0;
        }

        exposed_vertex_area += (vertex_area - temp_exposed_vertex_area) * (M_PI - angle) / (2.0 * M_PI);

        // Update positions for next iteration
        prev_pos_x = pos_x;
        prev_pos_y = pos_y;
        pos_x = next_pos_x;
        pos_y = next_pos_y;
    }

    particle_area[particle_id] = abs(temp_polygon_area) * 0.5 + exposed_vertex_area;
}

// overlap lenses
__global__ void kernelCalcParticleOverlapLenses(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, double* __restrict__ overlaps
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long num_neighbors = d_num_neighbors_ptr[particle_id];
    if (num_neighbors == 0) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];

    double overlap = 0.0;

    for (long n = 0; n < num_neighbors; n++) {
        long other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_id == -1 || other_id == particle_id) continue;

        double other_pos_x = positions_x[other_id];
        double other_pos_y = positions_y[other_id];
        double other_rad = radii[other_id];

        double dx = pbcDistance(pos_x, other_pos_x, 0);
        double dy = pbcDistance(pos_y, other_pos_y, 1);
        double rad_sum = rad + other_rad;
        double distance_sq = dx * dx + dy * dy;

        if (distance_sq >= rad_sum * rad_sum) {
            continue;
        }
        double distance = sqrt(distance_sq);
        // divide by 2 to avoid double counting
        overlaps[particle_id] += calcOverlapLenseArea(distance, rad, other_rad) / 2.0;
	}
}

// calculate particle positions

__global__ void kernelCalculateParticlePositions(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    double* __restrict__ particle_positions_x, double* __restrict__ particle_positions_y
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = 0.0;
    double pos_y = 0.0;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    // Loop over vertices to compute the area
    for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
        pos_x += vertex_positions_x[i];
        pos_y += vertex_positions_y[i];
    }

    particle_positions_x[particle_id] = pos_x / num_vertices;
    particle_positions_y[particle_id] = pos_y / num_vertices;
}



// vertex neighbors
__global__ void kernelUpdateVertexNeighborList(
    const double* __restrict__ vertex_positions_x, const double* __restrict__ vertex_positions_y,
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double cutoff,
    const double particle_cutoff
) {
    long vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id >= d_n_vertices) return;

    long added_vertex_neighbors = 0;
    double pos_x = vertex_positions_x[vertex_id];
    double pos_y = vertex_positions_y[vertex_id];
    double cutoff_sq = cutoff * cutoff;
    double particle_cutoff_sq = particle_cutoff * particle_cutoff;

    long particle_id = d_vertex_particle_index_ptr[vertex_id];
    long num_particle_neighbors = d_num_neighbors_ptr[particle_id];

    // iterate over neighboring particles in the particle neighbor list
    // printf("vertex_id: %ld - %ld: %ld / %ld\n", vertex_id, num_particle_neighbors, added_vertex_neighbors, d_max_vertex_neighbors_allocated);
    // printf("d_vertex_particle_index_ptr[%ld]: %ld\n", vertex_id, d_vertex_particle_index_ptr[vertex_id]);
    // printf("d_vertex_neighbor_list_ptr[%ld]: %ld\n", vertex_id, d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated]);
    // printf("d_max_vertex_neighbors_allocated: %ld\n", d_max_vertex_neighbors_allocated);
    
    for (long n = 0; n < num_particle_neighbors; n++) {
        long other_particle_id = d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + n];
        if (other_particle_id == -1 || other_particle_id == particle_id) continue;

        // Load neighbor's data
        double other_particle_x = positions_x[other_particle_id];
        double other_particle_y = positions_y[other_particle_id];

        // If the particle center is within the particle cutoff radius of the vertex, then check the vertices of the particle
        if (isWithinCutoffSquared(pos_x, pos_y, other_particle_x, other_particle_y, particle_cutoff_sq)) {
            long first_vertex_index = d_particle_start_index_ptr[other_particle_id];
            long num_vertices = d_num_vertices_in_particle_ptr[other_particle_id];

            for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
                double other_x = vertex_positions_x[i];
                double other_y = vertex_positions_y[i];

                if (isWithinCutoffSquared(pos_x, pos_y, other_x, other_y, cutoff_sq)) {
                    if (added_vertex_neighbors < d_max_vertex_neighbors_allocated) {
                        d_vertex_neighbor_list_ptr[vertex_id * d_max_vertex_neighbors_allocated + added_vertex_neighbors] = i;
                    }
                    added_vertex_neighbors++;
                }
            }
        }
    }
    d_num_vertex_neighbors_ptr[vertex_id] = added_vertex_neighbors;
    // printf("vertex_id: %ld - %ld\n", vertex_id, added_vertex_neighbors);
}


// scale positions

__global__ void kernelScalePositions(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ vertex_positions_x, double* __restrict__ vertex_positions_y,
    const double scale_factor
) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;


    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];

    double new_pos_x = pos_x * scale_factor;
    double new_pos_y = pos_y * scale_factor;

    positions_x[particle_id] = new_pos_x;
    positions_y[particle_id] = new_pos_y;

    long first_vertex_index = d_particle_start_index_ptr[particle_id];
    long num_vertices = d_num_vertices_in_particle_ptr[particle_id];

    for (long i = first_vertex_index; i < first_vertex_index + num_vertices; i++) {
        vertex_positions_x[i] += (new_pos_x - pos_x);
        vertex_positions_y[i] += (new_pos_y - pos_y);
    }
}
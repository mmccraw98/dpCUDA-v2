#include <stdio.h>
#include <cmath>
#include "../../include/constants.h"
#include "../../include/kernels/kernels.cuh"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

__constant__ long d_dim_block;
__constant__ long d_dim_grid;
__constant__ long d_dim_vertex_grid;

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
__constant__ long d_max_neighbors;
__constant__ long d_max_neighbors_allocated;

__constant__ long d_n_cells;
__constant__ long d_n_cells_dim;
__constant__ double d_cell_size;

// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------

// Should have the displacement calculation here rather than in velocities since this is only called once per time step
__global__ void kernelUpdatePositions(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    const double* __restrict__ last_neigh_positions_x, const double* __restrict__ last_neigh_positions_y,
    const double* __restrict__ last_cell_positions_x, const double* __restrict__ last_cell_positions_y,
    double* __restrict__ neigh_displacements_sq, double* __restrict__ cell_displacements_sq,
    double* __restrict__ velocities_x, double* __restrict__ velocities_y, const double dt) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    if (isnan(positions_x[particle_id]) || isnan(positions_y[particle_id])) {
        printf("NaN in positions for particle %ld: pos_x=%f, pos_y=%f\n", 
               particle_id, positions_x[particle_id], positions_y[particle_id]);
        positions_x[10000000000] += 10000000;
    }


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

    if (isnan(neigh_displacements_sq[particle_id])) {
        printf("NaN detected in displacement: dx=%f, dy=%f, particle_id=%ld\n", dx_neigh, dy_neigh, particle_id);
    }

    // Calculate squared displacement for cell list
    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;

    if (isnan(cell_displacements_sq[particle_id])) {
        printf("NaN detected in displacement: dx=%f, dy=%f, particle_id=%ld\n", dx_cell, dy_cell, particle_id);
    }

    if (cell_displacements_sq[particle_id] == 0.0) {
        // printf("Particle %ld: pos_x: %f, pos_y: %f, last_neigh_pos_x: %f, last_neigh_pos_y: %f, last_cell_pos_x: %f, last_cell_pos_y: %f\n", particle_id, pos_x, pos_y, last_neigh_positions_x[particle_id], last_neigh_positions_y[particle_id], last_cell_positions_x[particle_id], last_cell_positions_y[particle_id]);
    }
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


__global__ void kernelRemoveMeanVelocities(double* velocities) {
    long dim = threadIdx.x;
    if (dim < d_n_dim) {
        double velocity_avg = 0.0;
        for (long particle_id = 0; particle_id < d_n_particles; particle_id++) {
            velocity_avg += velocities[particle_id * d_n_dim + dim];
        }
        velocity_avg /= d_n_particles;
        for (long particle_id = 0; particle_id < d_n_particles; particle_id++) {
            velocities[particle_id * d_n_dim + dim] -= velocity_avg;
        }
    }
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


// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalcDiskForces(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    const double* __restrict__ radii, double* __restrict__ forces_x, 
    double* __restrict__ forces_y, double* __restrict__ potential_energy) 
{
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    double pos_x = positions_x[particle_id];
    double pos_y = positions_y[particle_id];
    double rad = radii[particle_id];
    double force_acc_x = 0.0, force_acc_y = 0.0;
    double energy = 0.0;

    for (long n = 0; n < d_num_neighbors_ptr[particle_id]; n++) {
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
        if (particle_id != other_id) {
            double other_x = positions_x[other_id];
            double other_y = positions_y[other_id];

            // Check if within cutoff using early exit
            if (isWithinCutoffSquared(pos_x, pos_y, other_x, other_y, cutoff_sq)) {
                if (added_neighbors < d_max_neighbors_allocated) {
                    d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
                }
                added_neighbors++;
            }
        }
    }
    last_neigh_positions_x[particle_id] = pos_x;
    last_neigh_positions_y[particle_id] = pos_y;
    neigh_displacements_sq[particle_id] = 0.0;
    d_num_neighbors_ptr[particle_id] = added_neighbors;
}


__global__ void kernelGetCellIndexForParticle(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y, 
    long* __restrict__ cell_index, long* __restrict__ sorted_cell_index, 
    long* __restrict__ particle_index) 
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
    sorted_cell_index[particle_id] = linear_cell_id;
    particle_index[particle_id] = particle_id;
}

__global__ void kernelGetFirstParticleIndexForCell(const long* sorted_cell_index, long* cell_start, const long width_offset, const long width) {
    long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= d_n_cells) return;

    if (cell_id == 0 && sorted_cell_index[0] == cell_id) {
        cell_start[cell_id] = 0;
        return;
    }

    // expand left
    long left = max(0L, (cell_id - width_offset) * width);
    while (left > 0 && sorted_cell_index[left] > cell_id) {
        left = max(0L, left - width);
    }

    // expand right
    long right = min(d_n_particles - 1, (cell_id + width_offset - 1) * width);
    while (right < d_n_particles - 1 && sorted_cell_index[right] < cell_id) {
        right = min(d_n_particles - 1, right + width);
    }

    // binary search to find an occurance of i
    long mid;
    bool found = false;
    while (left <= right) {
        mid = (left + right) / 2;
        if (sorted_cell_index[mid] == cell_id) {
            found = true;
            break;
        } else if (sorted_cell_index[mid] < cell_id) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if (found) {
        // find the leftmost occurance of cell_id using another binary search
        right = mid;
        while (sorted_cell_index[left] == cell_id) {
            left -= width;
        }

        while (left < right) {
            mid = (left + right) / 2;
            if (sorted_cell_index[mid] == cell_id) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        cell_start[cell_id] = left;
    }

    // other option
// __global__ void kernelGetFirstParticleIndexForCell(
//     const long* __restrict__ sorted_cell_index, 
//     long* __restrict__ cell_start, 
//     const long width_offset, 
//     const long width) 
// {
//     long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (cell_id >= d_n_cells) return;

//     // Default: No particles found for this cell
//     cell_start[cell_id] = -1;

//     // Calculate search bounds with clamping
//     long left = max(0L, (cell_id - width_offset) * width);
//     long right = min(d_n_particles - 1, (cell_id + width_offset - 1) * width);

//     // Binary search to find the leftmost occurrence of cell_id
//     while (left < right) {
//         long mid = (left + right) / 2;
//         long mid_value = sorted_cell_index[mid];  // Load into register

//         if (mid_value < cell_id) {
//             left = mid + 1;
//         } else {
//             right = mid;
//         }
//     }

//     // Verify if we found the cell_id at the left index
//     if (sorted_cell_index[left] == cell_id) {
//         cell_start[cell_id] = left;
//     }
// }
}


__global__ void kernelUpdateCellNeighborList(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    const double cutoff, const long* __restrict__ cell_index, 
    const long* __restrict__ particle_index, const long* __restrict__ cell_start, double* __restrict__ neigh_displacements_sq) 
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
            long end_id = cell_start[neighbor_cell_id + 1];

            if (start_id == -1) continue;  // No particles in this cell

            // Loop over particles in the neighboring cell
            for (long neighbor_id = start_id; neighbor_id < end_id; neighbor_id++) {
                long other_id = particle_index[neighbor_id];
                if (particle_id == other_id) continue;  // Skip self

                // Load neighbor particle positions directly
                double other_x = positions_x[other_id];
                double other_y = positions_y[other_id];

                // Calculate squared distance and apply cutoff check
                double dx = pos_x - other_x;
                double dy = pos_y - other_y;
                double dist_sq = dx * dx + dy * dy;

                if (dist_sq < cutoff_sq) {
                    if (added_neighbors < d_max_neighbors_allocated) {
                        d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
                    }
                    added_neighbors++;
                }
            }
        }
    }

    // Update the number of neighbors for this particle
    d_num_neighbors_ptr[particle_id] = added_neighbors;
    neigh_displacements_sq[particle_id] = 0.0;
    last_neigh_positions_x[particle_id] = pos_x;
    last_neigh_positions_y[particle_id] = pos_y;
}

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

    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    long index = particle_index[particle_id];

    temp_positions_x[index] = positions_x[particle_id];
    temp_positions_y[index] = positions_y[particle_id];
    temp_forces_x[index] = forces_x[particle_id];
    temp_forces_y[index] = forces_y[particle_id];
    temp_velocities_x[index] = velocities_x[particle_id];
    temp_velocities_y[index] = velocities_y[particle_id];
    temp_masses[index] = masses[particle_id];
    temp_radii[index] = radii[particle_id];
    last_cell_positions_x[index] = positions_x[particle_id];
    last_cell_positions_y[index] = positions_y[particle_id];
    cell_displacements_sq[index] = 0.0;
}
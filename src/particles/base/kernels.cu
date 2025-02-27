#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"
#include "../../../include/particles/base/kernels.cuh"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

__constant__ long d_particle_dim_block;
__constant__ long d_particle_dim_grid;
__constant__ long d_vertex_dim_grid;
__constant__ long d_vertex_dim_block;
__constant__ long d_cell_dim_grid;
__constant__ long d_cell_dim_block;

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

__global__ void kernelCalculateDampedForces(double* forces_x, double* forces_y, const double* velocities_x, const double* velocities_y, const double damping_coefficient) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    forces_x[particle_id] -= damping_coefficient * velocities_x[particle_id];
    forces_y[particle_id] -= damping_coefficient * velocities_y[particle_id];
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

__global__ void kernelStopRattlerVelocities(double* velocities_x, double* velocities_y, const long* __restrict__ contact_counts, const double rattler_threshold) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= d_n_particles) return;

    if (contact_counts[particle_id] < rattler_threshold) {
        velocities_x[particle_id] = 0.0;
        velocities_y[particle_id] = 0.0;
    }
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


// ----------------------------------------------------------------------
// --------------------------- Minimizers -------------------------------
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
// ---------------------------- Geometry --------------------------------
// ----------------------------------------------------------------------

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
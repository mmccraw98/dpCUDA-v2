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
    double* __restrict__ velocities_x, double* __restrict__ velocities_y, const double dt) 
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

    // Calculate squared displacement for cell list
    double dx_cell = pos_x - last_cell_positions_x[particle_id];
    double dy_cell = pos_y - last_cell_positions_y[particle_id];
    cell_displacements_sq[particle_id] = dx_cell * dx_cell + dy_cell * dy_cell;
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
// --------------------- Minimizers -------------------------------
// ----------------------------------------------------------------------

__global__ void kernelAdamStep(
    double* __restrict__ first_moment_x, double* __restrict__ first_moment_y,
    double* __restrict__ second_moment_x, double* __restrict__ second_moment_y,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double alpha, double beta1, double beta2, double one_minus_beta1_pow_t, 
    double one_minus_beta2_pow_t, double epsilon) {

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

    // Store updated moments back
    first_moment_x[particle_id] = first_m_x;
    first_moment_y[particle_id] = first_m_y;
    second_moment_x[particle_id] = second_m_x;
    second_moment_y[particle_id] = second_m_y;
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

// area

__global__ void kernelCalculateParticleArea(
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
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

__global__ void kernelUpdatePositions(double* positions, const double* last_positions, double* displacements, double* velocities, const double dt) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
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


__global__ void kernelCalculateTranslationalKineticEnergy(const double* velocities, const double* masses, double* kinetic_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        double velocity_sq = 0.0;
        #pragma unroll (N_DIM)
        for (long dim = 0; dim < d_n_dim; dim++) {
            velocity_sq += velocities[particle_id * d_n_dim + dim] * velocities[particle_id * d_n_dim + dim];
        }
        kinetic_energy[particle_id] = 0.5 * masses[particle_id] * velocity_sq;
    }
}

// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

__global__ void kernelCalcDiskForces(const double* positions, const double* radii, double* forces, double* potential_energy) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        long other_id;
        double this_pos[N_DIM], neighbor_pos[N_DIM];
        double this_rad, neighbor_rad;
        getPositionAndRadius(particle_id, positions, radii, this_pos, this_rad);
        for (long neighbor_id = 0; neighbor_id < d_num_neighbors_ptr[particle_id]; neighbor_id++) {
            if (isParticleNeighbor(particle_id, neighbor_id, other_id)) {
                getPositionAndRadius(other_id, positions, radii, neighbor_pos, neighbor_rad);
                potential_energy[particle_id] += calcPointPointInteraction(this_pos, neighbor_pos, this_rad + neighbor_rad, &forces[particle_id * d_n_dim]);
            }        
        }
    }
}


// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------

__global__ void kernelUpdateNeighborList(const double* positions, const double cutoff) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        long added_neighbors = 0;
        double this_pos[N_DIM], other_pos[N_DIM];
        double cutoff_sq = cutoff * cutoff;
        getPosition(particle_id, positions, this_pos);
        for (long other_id = 0; other_id < d_n_particles; other_id++) {
            if (particle_id != other_id) {
                getPosition(other_id, positions, other_pos);
                if (isWithinCutoffSquared(this_pos, other_pos, cutoff_sq)) {
                    if (added_neighbors < d_max_neighbors_allocated) {  // important for overflow concerns
                        d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
                    }
                    added_neighbors++;
                }
            }
        }
        d_num_neighbors_ptr[particle_id] = added_neighbors;
    }
}

__global__ void kernelGetCellIndexForParticle(const double* positions, long* cell_index, long* sorted_cell_index, long* particle_index) {
	long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        double this_pos[N_DIM];
        getPosition(particle_id, positions, this_pos);
        long x_index = getPBCCellIndex(this_pos[0]);
        long y_index = getPBCCellIndex(this_pos[1]);
        cell_index[particle_id] = x_index + y_index * d_n_cells_dim;
        sorted_cell_index[particle_id] = cell_index[particle_id];
        particle_index[particle_id] = particle_id;
    }
}

__global__ void kernelGetFirstParticleIndexForCell(const long* sorted_cell_index, long* cell_start, const long width_offset, const long width) {
    // this takes 10% of the gpu time
    // long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (cell_id < d_n_cells) {
    //     for (long particle_id = 0; particle_id < d_n_particles; particle_id++) {
    //         if (sorted_cell_index[particle_id] == cell_id) {
    //             cell_start[cell_id] = particle_id;
    //             break;
    //         }
    //         if (sorted_cell_index[particle_id] > cell_id) {
    //             cell_start[cell_id] = -1L;
    //             break;
    //         }
    //     }
    // }

    // binary search version
    long cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id < d_n_cells) {
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
    }
}


__global__ void kernelUpdateCellNeighborList(const double* positions, const double cutoff, const long* cell_index, const long* particle_index, const long* cell_start) {
    // old version
    // long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (particle_id < d_n_particles) {
    //     long added_neighbors = 0;
    //     double this_pos[N_DIM], other_pos[N_DIM];
    //     getPosition(particle_id, positions, this_pos);
    //     long cell_id = cell_index[particle_id];
    //     long cell_x = cell_id % d_n_cells_dim;
    //     long cell_y = cell_id / d_n_cells_dim;
    //     long start_id, end_id;
    //     for (long cell_y_offset = -1; cell_y_offset <= 1; cell_y_offset++) {
    //         for (long cell_x_offset = -1; cell_x_offset <= 1; cell_x_offset++) {
    //             long x_index = mod(cell_x + cell_x_offset, d_n_cells_dim);
    //             long y_index = mod(cell_y + cell_y_offset, d_n_cells_dim);
    //             long neighbor_cell_id = x_index + y_index * d_n_cells_dim;
    //             getCellIndexRange(neighbor_cell_id, cell_start, start_id, end_id);
    //             for (long neighbor_id = start_id; neighbor_id < end_id; neighbor_id++) {
    //                 long other_id = particle_index[neighbor_id];
    //                 if (particle_id != other_id) {
    //                     getPosition(other_id, positions, other_pos);
    //                     double distance = calcDistancePBC(this_pos, other_pos);
    //                     if (distance < cutoff) {
    //                         if (added_neighbors < d_max_neighbors_allocated) {  // important for overflow concerns
    //                             d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
    //                         }
    //                         added_neighbors++;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     d_num_neighbors_ptr[particle_id] = added_neighbors;
    // }


    // trying to be more efficient
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        long added_neighbors = 0;
        double this_pos[N_DIM], other_pos[N_DIM];
        getPosition(particle_id, positions, this_pos);
        long cell_id = cell_index[particle_id];
        long cell_x = cell_id % d_n_cells_dim;
        long cell_y = cell_id / d_n_cells_dim;
        double cutoff_sq = cutoff * cutoff;
        for (long cell_y_offset = -1; cell_y_offset <= 1; cell_y_offset++) {
            for (long cell_x_offset = -1; cell_x_offset <= 1; cell_x_offset++) {
                long x_index = mod(cell_x + cell_x_offset, d_n_cells_dim);
                long y_index = mod(cell_y + cell_y_offset, d_n_cells_dim);
                long neighbor_cell_id = x_index + y_index * d_n_cells_dim;

                // Get the index of the first particle in the neighbor cell
                long start_id = cell_start[neighbor_cell_id];
                if (start_id == -1) continue;  // no particles in the cell
                // Get the index of the last particle in the neighbor cell using the cell_start of the next cell
                long end_id = cell_start[neighbor_cell_id + 1];

                // Loop through all particles in the neighbor cell
                for (long neighbor_id = start_id; neighbor_id < end_id; neighbor_id++) {
                    long other_id = particle_index[neighbor_id];
                    if (particle_id != other_id) {
                        getPosition(other_id, positions, other_pos);
                        if (isWithinCutoffSquared(this_pos, other_pos, cutoff_sq)) {
                            if (added_neighbors < d_max_neighbors_allocated) {  // important for overflow concerns
                                d_neighbor_list_ptr[particle_id * d_max_neighbors_allocated + added_neighbors] = other_id;
                            }
                            added_neighbors++;
                        }
                    }
                }
            }
        }
        d_num_neighbors_ptr[particle_id] = added_neighbors;
    }

}
#pragma once

#include <stdio.h>
#include <cmath>
#include "../../../include/constants.h"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

extern __constant__ long d_particle_dim_block;  // number of threads per block
extern __constant__ long d_particle_dim_grid;  // number of blocks per grid
extern __constant__ long d_vertex_dim_grid;  // number of vertices per grid
extern __constant__ long d_vertex_dim_block;  // number of threads per block
extern __constant__ long d_cell_dim_grid;
extern __constant__ long d_cell_dim_block;

extern __constant__ double d_box_size[N_DIM];  // box size vector

extern __constant__ long d_n_dim;  // number of dimensions
extern __constant__ long d_n_particles;  // total number of particles
extern __constant__ long d_n_vertices;  // total number of vertices

extern __constant__ double d_e_c;  // energy scale for the contact energy
extern __constant__ double d_e_a;  // energy scale for the area energy
extern __constant__ double d_e_b;  // energy scale for the bending energy
extern __constant__ double d_e_l;  // energy scale for the length energy

extern __constant__ double d_n_c;  // exponent for the contact energy
extern __constant__ double d_n_a;  // exponent for the area energy
extern __constant__ double d_n_b;  // exponent for the bending energy
extern __constant__ double d_n_l;  // exponent for the length energy

extern __constant__ long* d_num_neighbors_ptr;  // pointer to the array that stores the number of neighbors for each particle
extern __constant__ long* d_neighbor_list_ptr;  // pointer to the neighbor list array
extern __constant__ long d_max_neighbors_allocated;  // maximum number of neighbors allocated for each particle

extern __constant__ long d_n_cells;  // number of cells in the simulation box
extern __constant__ long d_n_cells_dim;  // number of cells in each dimension
extern __constant__ double d_cell_size;  // size of the cells

// ----------------------------------------------------------------------
// -------------------------- General Purpose ---------------------------
// ----------------------------------------------------------------------

/**
 * @brief Modulo operation with non-negative result
 * 
 * @param a number to be moduloed
 * @param b modulo
 * @return __device__ non-negative result of the modulo operation
 */
inline __device__ long mod(long a, long b) {
    return (a % b + b) % b;  // Ensures non-negative result
}

/**
 * @brief X1 - X2 with periodic boundary conditions in the specified dimension
 * 
 * @param x1 position of the first vertex
 * @param x2 position of the second vertex
 * @param dim dimension for determining the box size
 * @return __device__ X1 - X2 - size * round((X1 - X2) / size)
 */
inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	double dist = x1 - x2, size = d_box_size[dim];
	return dist - size * round(dist / size); //round for distance, floor for position
    // double dist = x1 - x2;
    // double size = d_box_size[dim];
    // dist -= size * floor((dist + 0.5 * size) / size);  // Faster than round()
    // return dist;
}

// ----------------------------------------------------------------------
// ----------------------- Dynamics and Updates -------------------------
// ----------------------------------------------------------------------

/**
 * @brief Update the positions of the particles using an explicit Euler method.
 * Also updates the displacements of the particles from the last neighbor list update.
 * 
 * @param positions The positions of the particles.
 * @param last_positions The positions of the particles at the last time step.
 * @param displacements The displacements of the particles.
 * @param velocities The velocities of the particles.
 * @param dt The time step.
 */
__global__ void kernelUpdatePositions(double* positions_x, double* positions_y, const double* last_neigh_positions_x, const double* last_neigh_positions_y, const double* last_cell_positions_x, const double* last_cell_positions_y, double* neigh_displacements_sq, double* cell_displacements_sq, const double* velocities_x, const double* velocities_y, const double dt);


__global__ void kernelCalculateDampedForces(double* forces_x, double* forces_y, const double* velocities_x, const double* velocities_y, const double damping_coefficient);


__global__ void kernelMixVelocitiesAndForces(double* velocities_x, double* velocities_y, const double* forces_x, const double* forces_y, const double alpha);

/**
 * @brief Update the velocities of the particles using an explicit Euler method.
 * 
 * @param velocities The velocities of the particles.
 * @param forces The forces on the particles.
 * @param masses The masses of the particles.
 * @param dt The time step.
 */
__global__ void kernelUpdateVelocities(double* velocities_x, double* velocities_y, const double* forces_x, const double* forces_y, const double* masses, const double dt);


/**
 * @brief Removes the average velocity of the particles along a specified dimension.
 * 
 * @param velocities The velocities of the particles.
 */
__global__ void kernelRemoveMeanVelocities(double* __restrict__ velocities_x, double* __restrict__ velocities_y, const double mean_vel_x, const double mean_vel_y);


__global__ void kernelZeroForceAndPotentialEnergy(double* forces_x, double* forces_y, double* potential_energy);

/**
 * @brief Calculate the translational kinetic energy of the particles.
 * 
 * @param velocities The velocities of the particles.
 * @param masses The masses of the particles.
 * @param kinetic_energy The kinetic energy of the particles.
 */
__global__ void kernelCalculateTranslationalKineticEnergy(
    const double* __restrict__ velocities_x, const double* __restrict__ velocities_y,
    const double* __restrict__ masses, double* __restrict__ kinetic_energy);


__global__ void kernelStopRattlerVelocities(double* velocities_x, double* velocities_y, const long* __restrict__ contact_counts, const double rattler_threshold);

__global__ void kernelSetRandomCagePositions(double* positions_x, double* positions_y, const long* __restrict__ particle_cage_id, const long* __restrict__ cage_start_index, const double* __restrict__ cage_size_x, const double* __restrict__ cage_size_y, const double* __restrict__ random_numbers_x, const double* __restrict__ random_numbers_y, const double* __restrict__ cage_center_x, const double* __restrict__ cage_center_y);

__device__ __forceinline__ double tri_area(double ax, double ay, double bx, double by, double cx, double cy) {
    return 0.5 * fabs((bx - ax)*(cy - ay) - (cx - ax)*(by - ay));
}

__global__ void kernelSetRandomVoronoiPositions(double* positions_x, double* positions_y, const double* __restrict__ cage_center_x, const double* __restrict__ cage_center_y, const long* __restrict__ particle_cage_id, const long* __restrict__ cage_start_index, const double* __restrict__ voro_pos_x, const double* __restrict__ voro_pos_y, const long* __restrict__ voro_start_index, const long* __restrict__ voro_size, const double* __restrict__ voro_triangle_areas, const double* __restrict__ random_u, const double* __restrict__ random_v, const double* __restrict__ random_tri);

// ----------------------------------------------------------------------
// --------------------------- Interactions -----------------------------
// ----------------------------------------------------------------------

/**
 * @brief Calculate the result of the interaction between points 1 and 2 on point 1.  Adds the force to the force array.
 * Total energy of the interaction: V = e / n * (1 - r / sigma) ^ n
 * Energy for point 1 is 1/2 of the total.
 * 
 * @param point1 first point
 * @param point2 second point
 * @param rad_sum sum of the radii of the two points
 * @param force force on the first point
 * @return __device__ interaction energy between the two points
 */
inline __device__ double calcPointPointInteraction(
    const double pos_x, const double pos_y, const double rad,
    const double other_x, const double other_y, const double other_rad,
    double& force_x, double& force_y) 
{
    double dx = pbcDistance(pos_x, other_x, 0);
    double dy = pbcDistance(pos_y, other_y, 1);
    double rad_sum = rad + other_rad;
    double distance_sq = dx * dx + dy * dy;

    if (distance_sq >= rad_sum * rad_sum) {
        force_x = 0.0;
        force_y = 0.0;
        return 0.0;
    }

    double distance = sqrt(distance_sq);
    double overlap = 1.0 - (distance / rad_sum);
    double overlap_pow = pow(overlap, d_n_c - 1);

    // Calculate potential energy (half of it goes to point 1 and half to point 2)
    double energy = d_e_c * overlap * overlap_pow / (2.0 * d_n_c);

    // Calculate force magnitude and components
    double force_mag = d_e_c * overlap_pow / (rad_sum * distance);
    force_x = force_mag * dx;
    force_y = force_mag * dy;

    return energy;
}

inline __device__ double calcWallInteraction(const double pos_x, const double pos_y, const double rad, double& force_x, double& force_y) {
    double energy = 0.0;
    force_x = 0.0;
    force_y = 0.0;
    
    // Left wall
    if (pos_x < rad) {
        double dx = pos_x;  // distance from wall
        double overlap = rad - dx;
        if (overlap > 0) {
            double overlap_pow = pow(overlap / rad, d_n_c - 1);
            force_x += d_e_c * overlap_pow / rad;
            energy += d_e_c * overlap * overlap_pow / (d_n_c * rad);
        }
    }
    
    // Right wall
    if (pos_x > d_box_size[0] - rad) {
        double dx = d_box_size[0] - pos_x;  // distance from wall
        double overlap = rad - dx;
        if (overlap > 0) {
            double overlap_pow = pow(overlap / rad, d_n_c - 1);
            force_x -= d_e_c * overlap_pow / rad;
            energy += d_e_c * overlap * overlap_pow / (d_n_c * rad);
        }
    }
    
    // Bottom wall
    if (pos_y < rad) {
        double dy = pos_y;  // distance from wall
        double overlap = rad - dy;
        if (overlap > 0) {
            double overlap_pow = pow(overlap / rad, d_n_c - 1);
            force_y += d_e_c * overlap_pow / rad;
            energy += d_e_c * overlap * overlap_pow / (d_n_c * rad);
        }
    }
    
    // Top wall
    if (pos_y > d_box_size[1] - rad) {
        double dy = d_box_size[1] - pos_y;  // distance from wall
        double overlap = rad - dy;
        if (overlap > 0) {
            double overlap_pow = pow(overlap / rad, d_n_c - 1);
            force_y -= d_e_c * overlap_pow / rad;
            energy += d_e_c * overlap * overlap_pow / (d_n_c * rad);
        }
    }
    
    return energy;
}

inline __device__ double calcTorque(double force_x, double force_y, double pos_x, double pos_y, double center_x, double center_y) {
    double dx = pos_x - center_x;
    double dy = pos_y - center_y;
    return force_y * dx - force_x * dy;
}


// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------


inline __device__ bool isWithinCutoffSquared(
    double pos1_x, double pos1_y, 
    double pos2_x, double pos2_y, 
    double cutoff_sq) 
{
    double dx = pbcDistance(pos1_x, pos2_x, 0);
    double dy = pbcDistance(pos1_y, pos2_y, 1);
    double dist_sq = dx * dx + dy * dy;

    // Early exit if the distance already exceeds cutoff
    return dist_sq < cutoff_sq;
}


/**
 * @brief Update the neighbor list for all the particles.
 * 
 * @param positions The positions of the particles.
 * @param cutoff The cutoff distance for the neighbor list.
 */
__global__ void kernelUpdateNeighborList(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y, 
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y,
    double* __restrict__ neigh_displacements_sq,
    const double cutoff);

__global__ void kernelUpdateReplicaNeighborList(const long* __restrict__ voronoi_cell_size, const long* __restrict__ cage_start_id, const long* __restrict__ cage_size, const long max_cage_size);

/**
 * @brief Get the PBC cell index for a particle
 * 
 * @param pos position of the particle
 * @param cell_size size of the cells
 * @param n_cells_dim number of cells in each dimension
 * @return __device__ PBC cell index for the particle
 */
inline __device__ long getPBCCellIndex(double pos) {
    // version 3:
    // Use CUDA intrinsic for fast floor-like rounding
    long index = __double2ll_rd(pos / d_cell_size);
    // Ensure positive wrapping using optimized modulo
    return index >= 0 ? index % d_n_cells_dim : (index % d_n_cells_dim + d_n_cells_dim) % d_n_cells_dim;

    // version 2:
    // Calculate the index with floor-like rounding
    // long index = static_cast<long>(floor(pos / d_cell_size));
    // Ensure positive wrapping using modulo
    // return (index % d_n_cells_dim + d_n_cells_dim) % d_n_cells_dim;

    // version 1:
    // long index = __double2ll_rd(pos / d_cell_size);  // Efficient floor operation
    // return (index + d_n_cells_dim) % d_n_cells_dim;  // Ensure positive wrapping
}

/**
 * @brief Get the cell index for a particle
 * 
 * @param positions pointer to the array of positions of the particles
 * @param cell_index pointer to the array of cell indices of the particles
 * @param sorted_cell_index pointer to the array of cell indices of the particles which will eventually be sorted in ascending cell id order
 */
__global__ void kernelGetCellIndexForParticle(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    long* __restrict__ cell_index, long* __restrict__ particle_index);

/**
 * @brief Get the first particle index for each cell
 * 
 * @param sorted_cell_index pointer to the array of cell indices of the particles sorted in ascending cell id order
 * @param cell_start pointer to the array of first particle indices for each cell
 */
__global__ void kernelGetFirstParticleIndexForCell(const long* cell_index, long* cell_start, const long width_offset, const long width);


__global__ void kernelUpdateCellNeighborList(
    const double* __restrict__ positions_x, const double* __restrict__ positions_y,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y,
    const double cutoff, const long* __restrict__ cell_index, 
    const long* __restrict__ cell_start, double* __restrict__ cell_displacements_sq);

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
	double* __restrict__ cell_displacements_sq);


// ----------------------------------------------------------------------
// --------------------------- Minimizers -------------------------------
// ----------------------------------------------------------------------

__global__ void kernelAdamStep(
    double* __restrict__ first_moment_x, double* __restrict__ first_moment_y,
    double* __restrict__ second_moment_x, double* __restrict__ second_moment_y,
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    const double* __restrict__ forces_x, const double* __restrict__ forces_y,
    double alpha, double beta1, double beta2, double one_minus_beta1_pow_t, double one_minus_beta2_pow_t, double epsilon,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y, double* __restrict__ neigh_displacements_sq,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y, double* __restrict__ cell_displacements_sq);

__global__ void kernelGradDescStep(
    double* __restrict__ positions_x, double* __restrict__ positions_y,
    double* __restrict__ forces_x, double* __restrict__ forces_y,
    double* __restrict__ last_neigh_positions_x, double* __restrict__ last_neigh_positions_y, double* __restrict__ neigh_displacements_sq,
    double* __restrict__ last_cell_positions_x, double* __restrict__ last_cell_positions_y, double* __restrict__ cell_displacements_sq,
    double alpha);

// ----------------------------------------------------------------------
// ---------------------------- Geometry --------------------------------
// ----------------------------------------------------------------------

inline __device__ double angleBetweenVectors(const double next_x, const double next_y, const double current_x, const double current_y, const double previous_x, const double previous_y) {
    double mid_sin = (next_x - current_x) * (current_y - previous_y) - (next_y - current_y) * (current_x - previous_x);
    double mid_cos = (next_x - current_x) * (current_x - previous_x) + (next_y - current_y) * (current_y - previous_y);
    return atan2(mid_sin, mid_cos);
}

inline __device__ double A(const double R, const double d) {
    return R * R * acos(d / R) - d * sqrt(R * R - d * d);
}

inline __device__ double calcOverlapLenseArea(const double r_ij, const double radius_i, const double radius_j) {
    double d1 = (r_ij * r_ij - radius_i * radius_i + radius_j * radius_j) / (2 * r_ij);
    double d2 = r_ij - d1;
    return A(radius_i, d1) + A(radius_j, d2);
}

__global__ void kernelCalcPartcleOverlapLenses(const double* __restrict__ pos, const double* __restrict__ rad, double* __restrict__ overlaps);


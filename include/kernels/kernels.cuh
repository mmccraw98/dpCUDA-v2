#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <stdio.h>
#include <cmath>
#include "../../include/constants.h"

// ----------------------------------------------------------------------
// ----------------------- Device Constants -----------------------------
// ----------------------------------------------------------------------

extern __constant__ long d_dim_block;  // number of threads per block
extern __constant__ long d_dim_grid;  // number of blocks per grid
extern __constant__ long d_dim_vertex_grid;  // number of vertices per grid

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
extern __constant__ long d_max_neighbors;  // maximum number of neighbors
extern __constant__ long d_max_neighbors_allocated;  // maximum number of neighbors allocated for each particle


// ----------------------------------------------------------------------
// -------------------------- General Purpose ---------------------------
// ----------------------------------------------------------------------

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
}

/**
 * @brief Calculate the square of the L2 norm of a vector
 * 
 * @param segment vector to be squared
 * @return __device__ square of the L2 norm of the vector
 */
inline __device__ double calcNormSq(const double* segment) {
	double norm_sq = 0.;
	#pragma unroll (N_DIM)
	for (long dim = 0; dim < d_n_dim; dim++) {
		norm_sq += segment[dim] * segment[dim];
	}
	return norm_sq;
}

/**
 * @brief Calculate the L2 norm of an nDim-dimensional vector
 * 
 * @param segment vector of dimnesion nDim
 * @return __device__ L2 norm of the vector
 */
inline __device__ double calcNorm(const double* segment) {
    return sqrt(calcNormSq(segment));
}

/**
 * @brief Normalize a vector
 * 
 * @param segment vector to be normalized
 * @return __device__ norm of the vector
 */
inline __device__ double normalizeVector(double* segment) {
	double norm = calcNorm(segment);
	#pragma unroll (N_DIM)
	for (long dim = 0; dim < d_n_dim; dim++) {
		segment[dim] /= norm;
	}
	return norm;
}

/**
 * @brief Calculate the dot product of two vectors
 * 
 * @param segment1 first vector
 * @param segment2 second vector
 * @return __device__ dot product of the two vectors
 */
inline __device__ double dotProduct(const double* segment1, const double* segment2) {
	double dot_prod = 0.;
	#pragma unroll (N_DIM)
	for (long dim = 0; dim < d_n_dim; dim++) {
		dot_prod += segment1[dim] * segment2[dim];
	}
	return dot_prod;
}

/**
 * @brief Calculate the non-PBC distance vector (R_12) between segment1 and segment2 (R_12 = segment1 - segment2)
 * 
 * @param segment1 first segment
 * @param segment2 second segment
 * @param delta_vec delta_vec = segment1 - segment2 - overwrites the value of delta_vec
 */
inline __device__ void calcDelta(const double* segment1, const double* segment2, double* delta_vec) {
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	delta_vec[dim] = segment1[dim] - segment2[dim];
  	}
}

/**
 * @brief Calculate the non-PBC distance between two points
 * 
 * @param segment1 first point
 * @param segment2 second point
 * @return __device__ distance between the two points
 */
inline __device__ double calcDistance(const double* segment1, const double* segment2) {
	double dist_dim, distance_sq = 0.;
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	dist_dim = segment1[dim] - segment2[dim];
    	distance_sq += dist_dim * dist_dim;
  	}
  	return sqrt(distance_sq);
}

/**
 * @brief Calculate the non-PBC distance between two points and return the distance vector
 * 
 * @param segment1 first point
 * @param segment2 second point
 * @param delta_vec distance vector to be modified by the function (delta_vec = segment1 - segment2)
 * @return __device__ distance between the two points
 */
inline __device__ double calcDeltaAndDistance(const double* segment1, const double* segment2, double* delta_vec) {
	double dist_dim, distance_sq = 0.;
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	dist_dim = segment1[dim] - segment2[dim];
        delta_vec[dim] = dist_dim;
    	distance_sq += dist_dim * dist_dim;
  	}
  	return sqrt(distance_sq);
}

/**
 * @brief Calculate the PBC distance vector (R_12) between segment1 and segment2 (R_12 = segment1 - segment2) using periodic boundary conditions
 * 
 * @param segment1 Vector 1
 * @param segment2 Vector 2
 * @param delta_vec delta_vec = Vector 1 - Vector 2 in Periodic Boundary Conditions - overwrites the value of delta_vec
 */
inline __device__ void calcDeltaPBC(const double* segment1, const double* segment2, double* delta_vec) {
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	delta_vec[dim] = pbcDistance(segment1[dim], segment2[dim], dim);
  	}
}

/**
 * @brief Calculate the PBC distance between two points
 * 
 * @param segment1 first point
 * @param segment2 second point
 * @return __device__ distance between the two points
 */
inline __device__ double calcDistancePBC(const double* segment1, const double* segment2) {
  	double dist_dim, distance_sq = 0.;
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	dist_dim = pbcDistance(segment1[dim], segment2[dim], dim);
    	distance_sq += dist_dim * dist_dim;
  	}
  	return sqrt(distance_sq);
}

/**
 * @brief Calculate the PBC distance between two points and return the distance vector (R_12 = segment1 - segment2)
 * 
 * @param segment1 first point
 * @param segment2 second point
 * @param delta_vec distance vector to be modified by the function (delta_vec = segment1 - segment2)
 * @return __device__ distance between the two points
 */
inline __device__ double calcDeltaAndDistancePBC(const double* segment1, const double* segment2, double* delta_vec) {
	double dist_dim, distance_sq = 0.;
	#pragma unroll (N_DIM)
  	for (long dim = 0; dim < d_n_dim; dim++) {
    	dist_dim = pbcDistance(segment1[dim], segment2[dim], dim);
    	distance_sq += dist_dim * dist_dim;
    	delta_vec[dim] = dist_dim;
  	}
  	return sqrt(distance_sq);
}

/**
 * @brief Calculate the normal vector of a segment (90 degrees rotation)
 * 
 * @param segment segment
 * @param segment_normal normal vector of the segment
 */
inline __device__ void calcNormalVector(const double* segment, double* segment_normal) {
	segment_normal[0] = segment[1];
	segment_normal[1] = -segment[0];
}

/**
 * @brief Calculate the overlap between two points
 * 
 * @param point1 first point
 * @param point2 second point
 * @param rad_sum sum of the radii of the two points
 * @return __device__ overlap between the two points
 */
inline __device__ double calcOverlapPBC(const double* point1, const double* point2, const double rad_sum) {
	return (1 - calcDistancePBC(point1, point2) / rad_sum);
}

/**
 * @brief Calculate the overlap between two points and return the distance vector (R_12 = point1 - point2)
 * 
 * @param point1 first point
 * @param point2 second point
 * @param rad_sum sum of the radii of the two points
 * @param distance distance between the two points
 * @param delta_vec distance vector to be modified by the function (delta_vec = point1 - point2)
 * @return __device__ overlap between the two points
 */
inline __device__ double calcOverlapAndDeltaPBC(const double* point1, const double* point2, const double rad_sum, double* delta_vec, double& distance) {
    distance = calcDeltaAndDistancePBC(point1, point2, delta_vec);
    return (1 - distance / rad_sum);
}

/**
 * @brief Extract the position of a particle (or vertex) from the positions array and store it in pos
 * 
 * @param id particle id
 * @param positions positions of the particles
 * @param pos position of the particle
 */
inline __device__ void getPosition(const long id, const double* positions, double* pos) {
	#pragma unroll (N_DIM)
	for (long dim = 0; dim < d_n_dim; dim++) {
		pos[dim] = positions[id * d_n_dim + dim];
	}
}

/**
 * @brief Get the position and radius of a particle or vertex
 * 
 * @param id id of the particle or vertex
 * @param positions pointer to the array of positions
 * @param radii pointer to the array of radii
 * @param pos position of the particle or vertex
 * @param rad radius of the particle or vertex
 */
inline __device__ void getPositionAndRadius(const long id, const double* positions, const double* radii, double* pos, double& rad) {
    getPosition(id, positions, pos);
    rad = radii[id];
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
__global__ void kernelUpdatePositions(double* positions, const double* last_positions, double* displacements, double* velocities, const double dt);

/**
 * @brief Update the velocities of the particles using an explicit Euler method.
 * 
 * @param velocities The velocities of the particles.
 * @param forces The forces on the particles.
 * @param masses The masses of the particles.
 * @param dt The time step.
 */
__global__ void kernelUpdateVelocities(double* velocities, double* forces, const double* masses, const double dt);


// ----------------------------------------------------------------------
// --------------------------- Interactions -----------------------------
// ----------------------------------------------------------------------

/**
 * @brief Calculate the interaction between two points
 * V = e * (1 - r / sigma) ^ n
 * 
 * @param point1 first point
 * @param point2 second point
 * @param rad_sum sum of the radii of the two points
 * @param force force on the first point
 * @return __device__ interaction energy between the two points
 */
inline __device__ double calcPointPointInteraction(const double* point1, const double* point2, const double rad_sum, double* force) {
    double energy = 0.0;
    double distance, delta_vec[N_DIM];
    double overlap = calcOverlapAndDeltaPBC(point1, point2, rad_sum, delta_vec, distance);
    if (overlap > 0) {
        energy = d_e_c * pow(overlap, d_n_c) / d_n_c;
        #pragma unroll (N_DIM)
        for (long dim = 0; dim < d_n_dim; dim++) {
            force[dim] = d_e_c * pow(overlap, d_n_c - 1) * delta_vec[dim] / (rad_sum * distance);
        }
    }
    return energy;
}


// ----------------------------------------------------------------------
// ------------------------- Force Routines -----------------------------
// ----------------------------------------------------------------------

/**
 * @brief Calculate the interaction forces and energy between a particle and its neighbors.
 * First zeros out the potential energy of the particle, then sums up the potential energies of the interactions between the particle and its neighbors.
 * The initial zeroing may not be valid for certain particle types (i.e. DPM).
 * V = e / n * (1 - r / sigma) ^ n
 * 
 * @param positions Pointer to the array of positions of the particles.
 * @param radii Pointer to the array of radii of the particles.
 * @param forces Pointer to the array of forces on the particles.
 * @param potential_energy Pointer to the array of potential energies of the particles.
 */
__global__ void kernelCalcDiskForces(const double* positions, const double* radii, double* forces, double* potential_energy);


// ----------------------------------------------------------------------
// --------------------- Contacts and Neighbors -------------------------
// ----------------------------------------------------------------------

/**
 * @brief Check if a neighbor is a valid neighbor and get its true id
 * 
 * @param particle_id id of the particle
 * @param neighbor_id id of the neighbor
 * @param other_id the true id of the neighbor
 * @return __device__ true if the neighbor is a valid neighbor, false otherwise
 */
inline __device__ bool isParticleNeighbor(const long particle_id, const long neighbor_id, long& other_id) {
    other_id = d_neighbor_list_ptr[particle_id * d_max_neighbors + neighbor_id];
    return (particle_id != other_id && other_id != -1);
}

/**
 * @brief Update the neighbor list for all the particles.
 * 
 * @param positions The positions of the particles.
 * @param cutoff The cutoff distance for the neighbor list.
 */
__global__ void kernelUpdateNeighborList(const double* positions, const double cutoff);


#endif /* KERNELS_CUH_ */
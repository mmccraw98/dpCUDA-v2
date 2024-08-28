// include/kernels/UtilityKernels.cuh
#ifndef UTILITYKERNELS_CUH
#define UTILITYKERNELS_CUH

#include "CudaConstants.cuh"
#include "../Constants.h"

/**
 * @brief X1 - X2 with periodic boundary conditions in the specified dimension
 * 
 * @param x1 position of the first vertex
 * @param x2 position of the second vertex
 * @param dim dimension for determining the box size
 * @return __device__ X1 - X2 - size * round((X1 - X2) / size)
 */
inline __device__ double pbcDistance(const double x1, const double x2, const long dim) {
	double dist = x1 - x2, size = d_boxSizePointer[dim];
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
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
	#pragma unroll (MAXDIM)
  	for (long dim = 0; dim < d_nDim; dim++) {
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
 * @param delta_vec distance vector to be modified by the function (delta_vec = point1 - point2)
 * @return __device__ overlap between the two points
 */
inline __device__ double calcOverlapAndDeltaPBC(const double* point1, const double* point2, const double rad_sum, double* delta_vec) {
    return (1 - calcDeltaAndDistancePBC(point1, point2, delta_vec) / rad_sum);
}

#endif // UTILITYKERNELS_CUH
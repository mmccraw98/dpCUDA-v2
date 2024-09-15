
#ifndef CUDA_CONSTANTS_CUH_
#define CUDA_CONSTANTS_CUH_

#include "constants.h"
#include <stdio.h>

__constant__ long d_dim_block;  // number of threads per block
__constant__ long d_dim_grid;  // number of blocks per grid
__constant__ long d_dim_vertex_grid;  // number of vertices per grid

__constant__ double d_box_size[N_DIM];  // box size vector

__constant__ long d_n_dim = N_DIM;  // number of dimensions
__constant__ long d_n_particles;  // total number of particles
__constant__ long d_n_vertices;  // total number of vertices

__constant__ double d_e_c;  // energy scale for the contact energy
__constant__ double d_e_a;  // energy scale for the area energy
__constant__ double d_e_b;  // energy scale for the bending energy
__constant__ double d_e_l;  // energy scale for the length energy

__constant__ double d_n_c;  // exponent for the contact energy
__constant__ double d_n_a;  // exponent for the area energy
__constant__ double d_n_b;  // exponent for the bending energy
__constant__ double d_n_l;  // exponent for the length energy

__constant__ long* d_num_neighbors_ptr;  // pointer to the array that stores the number of neighbors for each particle
__constant__ long* d_neighbor_list_ptr;  // pointer to the neighbor list array
__constant__ long d_max_neighbors;  // maximum number of neighbors
__constant__ long d_max_neighbors_allocated;  // maximum number of neighbors allocated for each particle

#endif /* CUDA_CONSTANTS_CUH_ */
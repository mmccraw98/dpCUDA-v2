#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "../../include/constants.h"
#include "../../include/cuda_constants.cuh"
#include "../../include/kernels/general.cuh"
#include "../../include/kernels/contacts.cuh"

__global__ void kernelUpdateNeighborList(const double* positions, const double cutoff) {
    long particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id < d_n_particles) {
        printf("Particle %ld\n", particle_id);
        long added_neighbors = 0;
        double this_pos[N_DIM], other_pos[N_DIM];
        getPosition(particle_id, positions, this_pos);
        for (long other_id = 0; other_id < d_n_particles; other_id++) {
            if (particle_id != other_id) {
                getPosition(other_id, positions, other_pos);
                double distance = calcDistancePBC(this_pos, other_pos);
                if (distance < cutoff) {
                    d_neighbor_list_ptr[particle_id * d_max_neighbors + added_neighbors] = other_id;
                    added_neighbors++;
                }
            }
        }
        d_num_neighbors_ptr[particle_id] = added_neighbors;
    }
}
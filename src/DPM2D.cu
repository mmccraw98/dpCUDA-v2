// FUNCTION DECLARATIONS

#include "../include/constants.h"
#include "../include/DPM2D.h"
#include "../include/cuda_constants.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

DPM2D::DPM2D(long nParticles, long dim, long nVertexPerParticle, long randomSeed) {
	if (randomSeed == -1) {
		srand48(time(0));
	} else {
		srand48(randomSeed);
	}
	// default values
}

DPM2D::~DPM2D() {

}

void DPM2D::initializeBox(double area) {
    double box_size[N_DIM];
    double side_length = std::pow(area, 1.0 / N_DIM);
    for (int i = 0; i < N_DIM; i++) {
        box_size[i] = side_length;
    }
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_box_size, box_size, sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error copying box size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

thrust::host_vector<double> DPM2D::getBoxSize() {
    thrust::host_vector<double> box_size(N_DIM);
    cudaError_t cuda_err = cudaMemcpyFromSymbol(&box_size[0], d_box_size, sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error copying box size to host: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return box_size;
}

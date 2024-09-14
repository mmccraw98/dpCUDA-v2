#ifndef FUNCTORS_H_
#define FUNCTORS_H_

#include "constants.h"
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * @brief Functor to generate a random uniform number given a minimum, maximum, and seed
 * 
 */
struct RandomUniform {
    double min, max;
    unsigned int seed;

    RandomUniform(double min, double max, unsigned int seed) : min(min), max(max), seed(seed) {}

    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng(seed + n);
        thrust::uniform_real_distribution<double> dist(min, max);
        rng.discard(n);
        return dist(rng);
    }
};


/**
 * @brief Functor to generate a random normal number given a mean, standard deviation, and seed
 * 
 */
struct RandomNormal {
    double mean, stddev;
    unsigned int seed;

    RandomNormal(double mean, double stddev, unsigned int seed) : mean(mean), stddev(stddev), seed(seed) {}

    __host__ __device__
    double operator()(const unsigned int n) const {
        thrust::default_random_engine rng(seed);
        rng.discard(n);
        thrust::normal_distribution<double> dist(mean, stddev);
        return dist(rng);
    }
};
    
/**
 * @brief Functor to square a number
 * 
 */
struct Square {
    __host__ __device__ double operator()(const double x) {return x * x;}
};

/**
 * @brief Functor to compute the radius from the area of a circle
 * 
 */
struct RadiusFromArea {
    __host__ __device__ double operator()(const double area) { return sqrt(area / PI); }
};

/**
 * @brief Functor to compute the squared difference between a number from a set and its mean
 * 
 */
struct SquaredDifference {
    double mean;

    SquaredDifference(double mean) : mean(mean) {}

    __host__ __device__ double operator()(const double x) { return (x - mean) * (x - mean); }
};

#endif /* FUNCTORS_H_ */
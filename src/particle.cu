#include "../include/particle.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// Constructor
Particle::Particle() {
    d_boxSize.resize(MAXDIM);  // Resize box size as needed
    d_data.resize(5);  // Initialize device vector with size 5
}

// Destructor
Particle::~Particle() {
    d_boxSize.clear();
    d_data.clear();
}

// Multiplies device vector by a given factor
void Particle::multiplyData(double factor) {
    thrust::transform(d_data.begin(), d_data.end(), thrust::make_constant_iterator(factor), d_data.begin(), thrust::multiplies<double>());
}

// Virtual function to resize data to size 10
void Particle::resizeData() {
    d_data.resize(10);
}

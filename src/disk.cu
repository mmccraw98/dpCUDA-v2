#include "../include/disk.h"
#include <thrust/transform.h>
#include <thrust/device_vector.h>

// Constructor
Disk::Disk() : Particle() {
    // Can initialize specific properties for Disk if needed
}

// Destructor
Disk::~Disk() {
    // Clean up if necessary
}

// Override the unimplemented function: divides the device vector by a number
void Disk::unimplementedFunction() {
    double divisor = 2.0;  // Example divisor
    thrust::transform(d_data.begin(), d_data.end(), thrust::make_constant_iterator(1 / divisor), d_data.begin(), thrust::multiplies<double>());
}

// Override the resizeData function to resize the vector to size 15
void Disk::resizeData() {
    d_data.resize(15);
}

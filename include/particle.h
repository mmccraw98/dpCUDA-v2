#ifndef PARTICLE_H
#define PARTICLE_H

#include "defs.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Particle {
public:
    Particle();  // Constructor
    virtual ~Particle();  // Virtual destructor for proper cleanup

    // Member variable
    thrust::device_vector<double> d_data;

    // Member functions
    void multiplyData(double factor);  // Function to multiply device vector
    virtual void resizeData();  // Resizes vector, virtual for overriding
    virtual void unimplementedFunction() = 0;  // Pure virtual function

protected:
    thrust::device_vector<double> d_boxSize;
};

#endif // PARTICLE_H

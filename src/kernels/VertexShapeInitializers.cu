// src/kernels/VertexShapeInitializers.cu
#include "kernels/VertexShapeInitializers.cuh"

// Ideas:
// Store script parameters (SCRIPT VERSION NUMBER TOO!) whenever a run is made
// An object for each particle type - as a subclass of a general abstract class that the scripts will be designed around
// A configuration file for each particle type (for creating and loading) (maybe json :) )
// Wrap vectors in a class that handles dimensionality (can make single functions for loading / saving, validating sizes, etc.)
// Define grid, thread, and block dimensions intelligently

// Two types of kernels:
// Ones that act on a single particle(vertex)
// Ones that act on a set of vertices belonging to a single particle

// These will be the second type

// Should also move away from the static-script paradigm and move towards more dynamic, function based programming
// this should allow for faster (albeit more frequent) creation of one-off scripts which may be called from python

__device__ void putVerticesOnCircles(double* vertex_positions, double* particle_positions, const double* circle_radii) {
    long particle_id = blockIdx.x;
}
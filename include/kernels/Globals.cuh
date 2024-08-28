// include/kernels/Globals.cuh
#ifndef GLOBALS_CUH
#define GLOBALS_CUH

// Constants are defined globally on the device - all threads can access them - they are updated using cudaMemcpy

// Global variables or constants
__global__ double* d_boxSizePointer;

#endif // GLOBALS_CUH
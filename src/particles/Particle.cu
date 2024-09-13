// src/particles/Particle.cu

#include <iostream>
#include <unordered_map>
#include <any>
#include <typeinfo>
#include <thrust/device_vector.h>
#include "particles/Particle.cuh"
#include "../include/kernels/CudaConstants.cuh"

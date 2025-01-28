#pragma once

#include <typeinfo>
#include <iostream>
#include <cstdlib>
#include <thrust/device_vector.h>

template <typename T>
void printType(const T& obj) {
    std::cout << "Type: " << typeid(obj).name() << std::endl;
}

#define CUDA_CHECK(call) {cudaError_t err = call; if (err != cudaSuccess) {std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; std::exit(EXIT_FAILURE);}}
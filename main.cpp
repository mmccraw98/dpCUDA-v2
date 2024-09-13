// main.cpp
#include <iostream>
#include "particles/Disk.cuh"

int main() {
    std::cout << "Minimal CUDA Thrust and C++ Project" << std::endl;
    Disk disk(100, 0);

    // disk.initializeBox(1.0);

    return 0;
}
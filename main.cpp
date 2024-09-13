// main.cpp
#include <iostream>
#include "particles/Disk.cuh"

int main() {
    std::cout << "Minimal CUDA Thrust and C++ Project" << std::endl;
    Disk disk(100, 0);
    disk.updatePositions(0.1);
    disk.updateMomenta(0.1);
    return 0;
}
// main.cpp
#include <iostream>
#include "particles/Disk.cuh"

int main() {
    std::cout << "asdfasdfasdfMinimal CUDA Thrust and C++ Project" << std::endl;
    Disk disk(100, 0);
    disk.calculateForces();
    disk.updatePositions(0.1);
    disk.updateMomenta(0.1);

    std::cout << "Done" << std::endl;
    return 0;
}
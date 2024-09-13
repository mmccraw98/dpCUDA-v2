// main.cpp
#include <iostream>
#include "particles/Disk.cuh"

int main() {
    std::cout << "asdfasdfasdfMinimal CUDA Thrust and C++ Project" << std::endl;
    Disk disk(100, 0);
    disk.calculateForces();
    disk.updatePositions(0.1);
    disk.updateMomenta(0.1);

    thrust::host_vector<double> host_positions = disk.getArray<double>("d_positions");
    for (int i = 0; i < 10; i++) {
        std::cout << host_positions[i] << std::endl;
    }

    std::cout << "Done" << std::endl;
    return 0;
}
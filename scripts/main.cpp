#include "../include/particle/disk.h"

int main() {
    std::cout << "Minimal CUDA Thrust and C++ Project" << std::endl;

    Disk disk(100, 0);
    disk.initializeBox(1.0);
    thrust::host_vector<double> box_size = disk.getBoxSize();
    std::cout << "Box size: " << box_size[0] << ", " << box_size[1] << std::endl;
    return 0;
}

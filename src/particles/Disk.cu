// src/particles/Disk.cu
#include "particles/Disk.cuh"

Disk::Disk(long n_particles, long seed) {
    std::cout << "Disk constructor" << std::endl;
}

Disk::~Disk() {
    std::cout << "Disk destructor" << std::endl;
}

void Disk::updatePositionsImpl(double dt) {
    std::cout << "Updating positions" << std::endl;
}

void Disk::updateMomentaImpl(double dt) {
    std::cout << "Updating momenta" << std::endl;
}

void Disk::calculateForcesImpl() {
    std::cout << "Calculating forces" << std::endl;
}
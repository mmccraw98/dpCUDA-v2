// src/particles/Disk.cu
#include "particles/Disk.cuh"

Disk::Disk(long n_particles, long seed) {
    std::cout << "Disk::Disk" << std::endl;
}

Disk::~Disk() {
    std::cout << "Disk::~Disk" << std::endl;
}

void Disk::updatePositionsImpl(double dt) {
    std::cout << "Disk::updatePositionsImpl" << std::endl;
}

void Disk::updateMomentaImpl(double dt) {
    std::cout << "Disk::updateMomentaImpl" << std::endl;
}

void Disk::calculateForcesImpl() {
    std::cout << "Disk::calculateForcesImpl" << std::endl;
}
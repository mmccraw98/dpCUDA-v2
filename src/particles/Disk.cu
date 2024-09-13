// src/particles/Disk.cu
#include "particles/Disk.cuh"

Disk::Disk(long n_particles, long seed) {
    std::cout << "Disk::Disk" << std::endl;
    this->n_particles = n_particles;
    this->n_dim = 2;

    d_positions.resize(n_particles * n_dim);
    d_momenta.resize(n_particles * n_dim);
    d_forces.resize(n_particles * n_dim);
    d_radii.resize(n_particles);
    d_masses.resize(n_particles);
    d_potential_energy.resize(n_particles);
    d_kinetic_energy.resize(n_particles);
    d_last_positions.resize(n_particles * n_dim);
    d_neighbor_list.resize(n_particles);
    d_box_size.resize(n_dim);

    d_test_array.resize(n_particles);
}

Disk::~Disk() {
    std::cout << "Disk::~Disk" << std::endl;
    d_positions.clear();
    d_momenta.clear();
    d_forces.clear();
    d_radii.clear();
    d_masses.clear();
    d_potential_energy.clear();
    d_kinetic_energy.clear();
    d_last_positions.clear();
    d_neighbor_list.clear();
    d_box_size.clear();

    d_test_array.clear();
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
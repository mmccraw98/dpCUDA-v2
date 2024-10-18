#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/kernels/kernels.cuh"
#include "../../include/particle/config.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << " at line "        \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    }

Particle::Particle() {
}

Particle::~Particle() {
    clearDynamicVariables();
    clearGeometricVariables();
}

void Particle::initializeFromConfig(const BaseParticleConfig& config) {
}

// ----------------------------------------------------------------------
// ----------------------- Template Methods -----------------------------
// ----------------------------------------------------------------------

std::unordered_map<std::string, std::any> Particle::getArrayMap() {
    std::unordered_map<std::string, std::any> array_map;
    array_map["d_positions_x"]          = &d_positions_x;
    array_map["d_positions_y"]          = &d_positions_y;
    array_map["d_last_neigh_positions_x"]     = &d_last_neigh_positions_x;
    array_map["d_last_neigh_positions_y"]     = &d_last_neigh_positions_y;
    array_map["d_last_cell_positions_x"]     = &d_last_cell_positions_x;
    array_map["d_last_cell_positions_y"]     = &d_last_cell_positions_y;
    array_map["d_neigh_displacements_sq"]      = &d_neigh_displacements_sq;
    array_map["d_cell_displacements_sq"]      = &d_cell_displacements_sq;
    array_map["d_velocities_x"]         = &d_velocities_x;
    array_map["d_velocities_y"]         = &d_velocities_y;
    array_map["d_forces_x"]             = &d_forces_x;
    array_map["d_forces_y"]             = &d_forces_y;
    array_map["d_radii"]              = &d_radii;
    array_map["d_masses"]             = &d_masses;
    array_map["d_potential_energy"]   = &d_potential_energy;
    array_map["d_kinetic_energy"]     = &d_kinetic_energy;
    array_map["d_neighbor_list"]      = &d_neighbor_list;
    array_map["d_num_neighbors"]      = &d_num_neighbors;
    array_map["d_cell_index"]         = &d_cell_index;
    array_map["d_sorted_cell_index"]  = &d_sorted_cell_index;
    array_map["d_particle_index"]     = &d_particle_index;
    array_map["d_cell_start"]         = &d_cell_start;
    return array_map;
}

std::string Particle::getArrayType(const std::string& array_name) {
    std::unordered_map<std::string, std::string> array_type_map;
    array_type_map["d_positions_x"]          = "double";
    array_type_map["d_positions_y"]          = "double";
    array_type_map["d_last_neigh_positions_x"]     = "double";
    array_type_map["d_last_neigh_positions_y"]     = "double";
    array_type_map["d_last_cell_positions_x"]     = "double";
    array_type_map["d_last_cell_positions_y"]     = "double";
    array_type_map["d_neigh_displacements_sq"]      = "double";
    array_type_map["d_cell_displacements_sq"]      = "double";
    array_type_map["d_velocities_x"]         = "double";
    array_type_map["d_velocities_y"]         = "double";
    array_type_map["d_forces_x"]             = "double";
    array_type_map["d_forces_y"]             = "double";
    array_type_map["d_radii"]              = "double";
    array_type_map["d_masses"]             = "double";
    array_type_map["d_potential_energy"]   = "double";
    array_type_map["d_kinetic_energy"]     = "double";
    array_type_map["d_box_size"]           = "double";
    array_type_map["d_neighbor_list"]      = "long";
    array_type_map["d_num_neighbors"]      = "long";
    array_type_map["d_cell_index"]         = "long";
    array_type_map["d_sorted_cell_index"]  = "long";
    array_type_map["d_particle_index"]     = "long";
    array_type_map["d_cell_start"]         = "long";
    return array_type_map[array_name];
}

// ----------------------------------------------------------------------
// -------------------- Universally Defined Methods ---------------------
// ----------------------------------------------------------------------

void Particle::setNeighborListUpdateMethod(std::string method_name) {
    if (method_name == "cell") {
        std::cout << "Particle::setNeighborListUpdateMethod: Setting neighbor list update method to cell" << std::endl;
        this->updateNeighborListPtr = &Particle::updateCellNeighborList;
        this->checkForNeighborUpdatePtr = &Particle::checkForCellUpdate;
    } else if (method_name == "verlet") {
        std::cout << "Particle::setNeighborListUpdateMethod: Setting neighbor list update method to verlet" << std::endl;
        this->updateNeighborListPtr = &Particle::updateNeighborList;
        this->checkForNeighborUpdatePtr = &Particle::checkForNeighborUpdate;
    } else if (method_name == "none") {
        std::cout << "Particle::setNeighborListUpdateMethod: Setting neighbor list update method to none" << std::endl;
        throw std::invalid_argument("Particle::setNeighborListUpdateMethod: 'none' neighbor list update method not implemented: " + method_name);
    } else {
        throw std::invalid_argument("Particle::setNeighborListUpdateMethod: Invalid method name: " + method_name);
    }
}

void Particle::setSeed(long seed) {
    if (seed == -1) {
        seed = time(0);
    }
    this->seed = seed;
    srand(seed);
}

void Particle::setNumParticles(long n_particles) {
    this->n_particles = n_particles;
    syncNumParticles();
}

void Particle::syncNumParticles() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_n_particles, &n_particles, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNumParticles: Error copying n_particles to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::setDegreesOfFreedom() {
    this->n_dof = n_particles * N_DIM;
}

void Particle::setNumVertices(long n_vertices) {
    this->n_vertices = n_vertices;
    syncNumVertices();
}

void Particle::syncNumVertices() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_n_vertices, &n_vertices, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNumVertices: Error copying n_vertices to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::setParticleCounts(long n_particles, long n_vertices) {
    setNumParticles(n_particles);
    setNumVertices(n_vertices);
    setDegreesOfFreedom();
    initDynamicVariables();
    initGeometricVariables();
}

void Particle::setKernelDimensions(long dim_block) {
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "CUDA Info: Particle::setKernelDimensions: Max threads per block: " << maxThreadsPerBlock << std::endl;
    if (dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: Particle::setKernelDimensions: dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        dim_block = maxThreadsPerBlock;
    }
    this->dim_block = dim_block;
    // Implement some particle-specific logic to define the grid dimensions
    // Then, sync
    std::cout << "WARNING: Particle::setKernelDimensions: Not Implemented" << std::endl;
    syncKernelDimensions();
}

void Particle::syncKernelDimensions() {
    cudaError_t cuda_err;
    cuda_err = cudaMemcpyToSymbol(d_dim_block, &dim_block, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying dim_block to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_grid, &dim_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying dim_grid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_vertex_grid, &dim_vertex_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying dim_vertex_grid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::initDynamicVariables() {
    // Resize the device vectors
    d_positions_x.resize(n_particles);
    d_positions_y.resize(n_particles);
    d_last_neigh_positions_x.resize(n_particles);
    d_last_neigh_positions_y.resize(n_particles);
    d_last_cell_positions_x.resize(n_particles);
    d_last_cell_positions_y.resize(n_particles);
    d_neigh_displacements_sq.resize(n_particles);
    d_cell_displacements_sq.resize(n_particles);
    d_velocities_x.resize(n_particles);
    d_velocities_y.resize(n_particles);
    d_forces_x.resize(n_particles);
    d_forces_y.resize(n_particles);
    d_radii.resize(n_particles);
    d_masses.resize(n_particles);
    d_potential_energy.resize(n_particles);
    d_kinetic_energy.resize(n_particles);
    d_neighbor_list.resize(n_particles);
    d_num_neighbors.resize(n_particles);
    d_temp_positions_x.resize(n_particles);
    d_temp_positions_y.resize(n_particles);
    d_temp_forces_x.resize(n_particles);
    d_temp_forces_y.resize(n_particles);
    d_temp_velocities_x.resize(n_particles);
    d_temp_velocities_y.resize(n_particles);
    d_temp_masses.resize(n_particles);
    d_temp_radii.resize(n_particles);

    thrust::fill(d_positions_x.begin(), d_positions_x.end(), 0.0);
    thrust::fill(d_positions_y.begin(), d_positions_y.end(), 0.0);
    thrust::fill(d_last_neigh_positions_x.begin(), d_last_neigh_positions_x.end(), 0.0);
    thrust::fill(d_last_neigh_positions_y.begin(), d_last_neigh_positions_y.end(), 0.0);
    thrust::fill(d_last_cell_positions_x.begin(), d_last_cell_positions_x.end(), 0.0);
    thrust::fill(d_last_cell_positions_y.begin(), d_last_cell_positions_y.end(), 0.0);
    thrust::fill(d_neigh_displacements_sq.begin(), d_neigh_displacements_sq.end(), 0.0);
    thrust::fill(d_cell_displacements_sq.begin(), d_cell_displacements_sq.end(), 0.0);
    thrust::fill(d_velocities_x.begin(), d_velocities_x.end(), 0.0);
    thrust::fill(d_velocities_y.begin(), d_velocities_y.end(), 0.0);
    thrust::fill(d_forces_x.begin(), d_forces_x.end(), 0.0);
    thrust::fill(d_forces_y.begin(), d_forces_y.end(), 0.0);
    thrust::fill(d_radii.begin(), d_radii.end(), 0.0);
    thrust::fill(d_masses.begin(), d_masses.end(), 0.0);
    thrust::fill(d_potential_energy.begin(), d_potential_energy.end(), 0.0);
    thrust::fill(d_kinetic_energy.begin(), d_kinetic_energy.end(), 0.0);
    thrust::fill(d_temp_positions_x.begin(), d_temp_positions_x.end(), 0.0);
    thrust::fill(d_temp_positions_y.begin(), d_temp_positions_y.end(), 0.0);
    thrust::fill(d_temp_forces_x.begin(), d_temp_forces_x.end(), 0.0);
    thrust::fill(d_temp_forces_y.begin(), d_temp_forces_y.end(), 0.0);
    thrust::fill(d_temp_velocities_x.begin(), d_temp_velocities_x.end(), 0.0);
    thrust::fill(d_temp_velocities_y.begin(), d_temp_velocities_y.end(), 0.0);
    thrust::fill(d_temp_masses.begin(), d_temp_masses.end(), 0.0);
    thrust::fill(d_temp_radii.begin(), d_temp_radii.end(), 0.0);

    // max_neighbors = 0;
    // max_neighbors_allocated = 0;
    thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
    thrust::fill(d_num_neighbors.begin(), d_num_neighbors.end(), max_neighbors);


    // Cast the raw pointers
    d_positions_x_ptr = thrust::raw_pointer_cast(&d_positions_x[0]);
    d_positions_y_ptr = thrust::raw_pointer_cast(&d_positions_y[0]);
    d_last_neigh_positions_x_ptr = thrust::raw_pointer_cast(&d_last_neigh_positions_x[0]);
    d_last_neigh_positions_y_ptr = thrust::raw_pointer_cast(&d_last_neigh_positions_y[0]);
    d_last_cell_positions_x_ptr = thrust::raw_pointer_cast(&d_last_cell_positions_x[0]);
    d_last_cell_positions_y_ptr = thrust::raw_pointer_cast(&d_last_cell_positions_y[0]);
    d_neigh_displacements_sq_ptr = thrust::raw_pointer_cast(&d_neigh_displacements_sq[0]);
    d_cell_displacements_sq_ptr = thrust::raw_pointer_cast(&d_cell_displacements_sq[0]);
    d_velocities_x_ptr = thrust::raw_pointer_cast(&d_velocities_x[0]);
    d_velocities_y_ptr = thrust::raw_pointer_cast(&d_velocities_y[0]);
    d_forces_x_ptr = thrust::raw_pointer_cast(&d_forces_x[0]);
    d_forces_y_ptr = thrust::raw_pointer_cast(&d_forces_y[0]);
    d_radii_ptr = thrust::raw_pointer_cast(&d_radii[0]);
    d_masses_ptr = thrust::raw_pointer_cast(&d_masses[0]);
    d_potential_energy_ptr = thrust::raw_pointer_cast(&d_potential_energy[0]);
    d_kinetic_energy_ptr = thrust::raw_pointer_cast(&d_kinetic_energy[0]);
    d_temp_positions_x_ptr = thrust::raw_pointer_cast(&d_temp_positions_x[0]);
    d_temp_positions_y_ptr = thrust::raw_pointer_cast(&d_temp_positions_y[0]);
    d_temp_forces_x_ptr = thrust::raw_pointer_cast(&d_temp_forces_x[0]);
    d_temp_forces_y_ptr = thrust::raw_pointer_cast(&d_temp_forces_y[0]);
    d_temp_velocities_x_ptr = thrust::raw_pointer_cast(&d_temp_velocities_x[0]);
    d_temp_velocities_y_ptr = thrust::raw_pointer_cast(&d_temp_velocities_y[0]);
    d_temp_masses_ptr = thrust::raw_pointer_cast(&d_temp_masses[0]);
    d_temp_radii_ptr = thrust::raw_pointer_cast(&d_temp_radii[0]);
}

void Particle::clearDynamicVariables() {
    // Clear the device vectors
    d_positions_x.clear();
    d_positions_y.clear();
    d_last_neigh_positions_x.clear();
    d_last_neigh_positions_y.clear();
    d_neigh_displacements_sq.clear();
    d_cell_displacements_sq.clear();
    d_last_cell_positions_x.clear();
    d_last_cell_positions_y.clear();
    d_velocities_x.clear();
    d_velocities_y.clear();
    d_forces_x.clear();
    d_forces_y.clear();
    d_radii.clear();
    d_masses.clear();
    d_potential_energy.clear();
    d_kinetic_energy.clear();
    d_neighbor_list.clear();
    d_num_neighbors.clear();
    d_cell_index.clear();
    d_sorted_cell_index.clear();
    d_particle_index.clear();
    d_cell_start.clear();
    d_temp_positions_x.clear();
    d_temp_positions_y.clear();
    d_temp_forces_x.clear();
    d_temp_forces_y.clear();
    d_temp_velocities_x.clear();
    d_temp_velocities_y.clear();
    d_temp_masses.clear();
    d_temp_radii.clear();

    // Clear the pointers
    d_positions_x_ptr = nullptr;
    d_positions_y_ptr = nullptr;
    d_last_neigh_positions_x_ptr = nullptr;
    d_last_neigh_positions_y_ptr = nullptr;
    d_last_cell_positions_x_ptr = nullptr;
    d_last_cell_positions_y_ptr = nullptr;
    d_neigh_displacements_sq_ptr = nullptr;
    d_cell_displacements_sq_ptr = nullptr;
    d_velocities_x_ptr = nullptr;
    d_velocities_y_ptr = nullptr;
    d_forces_x_ptr = nullptr;
    d_forces_y_ptr = nullptr;
    d_radii_ptr = nullptr;
    d_masses_ptr = nullptr;
    d_potential_energy_ptr = nullptr;
    d_kinetic_energy_ptr = nullptr;
    d_cell_index_ptr = nullptr;
    d_sorted_cell_index_ptr = nullptr;
    d_particle_index_ptr = nullptr;
    d_cell_start_ptr = nullptr;
    d_temp_positions_x_ptr = nullptr;
    d_temp_positions_y_ptr = nullptr;
    d_temp_forces_x_ptr = nullptr;
    d_temp_forces_y_ptr = nullptr;
    d_temp_velocities_x_ptr = nullptr;
    d_temp_velocities_y_ptr = nullptr;
    d_temp_masses_ptr = nullptr;
    d_temp_radii_ptr = nullptr;
}

void Particle::setBoxSize(const thrust::host_vector<double>& box_size) {
    if (box_size.size() != N_DIM) {
        throw std::invalid_argument("Particle::setBoxSize: Error box_size (" + std::to_string(box_size.size()) + ")" + " != " + std::to_string(N_DIM) + " elements");
    }
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_box_size, box_size.data(), sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setBoxSize: Error copying box size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

thrust::host_vector<double> Particle::getBoxSize() {
    thrust::host_vector<double> box_size(N_DIM);
    cudaError_t cuda_err = cudaMemcpyFromSymbol(&box_size[0], d_box_size, sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::getBoxSize: Error copying box size to host: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return box_size;
}

void Particle::syncNeighborList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_neighbors_allocated, &this->max_neighbors_allocated, sizeof(this->max_neighbors_allocated));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying max_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    long* neighbor_list_ptr = thrust::raw_pointer_cast(&d_neighbor_list[0]);
    cuda_err = cudaMemcpyToSymbol(d_neighbor_list_ptr, &neighbor_list_ptr, sizeof(neighbor_list_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying d_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    long* num_neighbors_ptr = thrust::raw_pointer_cast(&d_num_neighbors[0]);
    cuda_err = cudaMemcpyToSymbol(d_num_neighbors_ptr, &num_neighbors_ptr, sizeof(num_neighbors_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying d_num_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::setEnergyScale(double e, std::string which) {
    cudaError_t cuda_err;
    if (which == "c") {
        e_c = e;
        cuda_err = cudaMemcpyToSymbol(d_e_c, &e_c, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_c to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "a") {
        e_a = e;
        cuda_err = cudaMemcpyToSymbol(d_e_a, &e_a, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_a to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "b") {
        e_b = e;
        cuda_err = cudaMemcpyToSymbol(d_e_b, &e_b, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_b to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "l") {
        e_l = e;
        cuda_err = cudaMemcpyToSymbol(d_e_l, &e_l, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_l to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        throw std::invalid_argument("Particle::setEnergyScale: which must be 'c', 'a', 'b', or 'l', not " + which);
    }
}

double Particle::getEnergyScale(std::string which) {
    if (which == "c") {
        return e_c;
    } else if (which == "a") {
        return e_a;
    } else if (which == "b") {
        return e_b;
    } else if (which == "l") {
        return e_l;
    } else {
        throw std::invalid_argument("Particle::getEnergyScale: which must be 'c', 'a', 'b', or 'l', not " + which);
    }
}

void Particle::setAllEnergyScales(double e_c, double e_a, double e_b, double e_l) {
    setEnergyScale(e_c, "c");
    setEnergyScale(e_a, "a");
    setEnergyScale(e_b, "b");
    setEnergyScale(e_l, "l");
}

void Particle::setExponent(double n, std::string which) {
    cudaError_t cuda_err;
    if (which == "c") {
        n_c = n;
        cuda_err = cudaMemcpyToSymbol(d_n_c, &n_c, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_c to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "a") {
        n_a = n;
        cuda_err = cudaMemcpyToSymbol(d_n_a, &n_a, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_a to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "b") {
        n_b = n;
        cuda_err = cudaMemcpyToSymbol(d_n_b, &n_b, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_b to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (which == "l") {
        n_l = n;
        cuda_err = cudaMemcpyToSymbol(d_n_l, &n_l, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_l to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        throw std::invalid_argument("Particle::setExponent: which must be 'c', 'a', 'b', or 'l', not " + which);
    }
}

void Particle::setAllExponents(double n_c, double n_a, double n_b, double n_l) {
    setExponent(n_c, "c");
    setExponent(n_a, "a");
    setExponent(n_b, "b");
    setExponent(n_l, "l");
}

double Particle::getExponent(std::string which) {
    if (which == "c") {
        return n_c;
    } else if (which == "a") {
        return n_a;
    } else if (which == "b") {
        return n_b;
    } else if (which == "l") {
        return n_l;
    } else {
        throw std::invalid_argument("Particle::getExponent: which must be 'c', 'a', 'b', or 'l', not " + which);
    }
}

void Particle::initializeBox(double packing_fraction) {
    // set the box size to an arbitrary initial value
    double side_length = 1.0;
    thrust::host_vector<double> box_size(N_DIM, side_length);
    setBoxSize(box_size);
    // then rescale the box size to the desired packing fraction
    scaleToPackingFraction(packing_fraction);
}

void Particle::setRandomUniform(thrust::device_vector<double>& values, double min, double max) {
    thrust::counting_iterator<long> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomUniform(min, max, seed));
}

void Particle::setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev) {
    std::cout << "Set: This does not work yet" << std::endl;
    thrust::counting_iterator<long> index_sequence_begin(seed);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomNormal(mean, stddev, seed));
}

void Particle::setRandomPositions() {
    thrust::host_vector<double> box_size = getBoxSize();
    setRandomUniform(d_positions_x, 0.0, box_size[0]);
    setRandomUniform(d_positions_y, 0.0, box_size[1]);
}

void Particle::removeMeanVelocities() {
    std::cout << "Remove: This does not work yet" << std::endl;
    // kernelRemoveMeanVelocities<<<1, N_DIM>>>(d_velocities_ptr);
    // cudaDeviceSynchronize();
}

void Particle::scaleVelocitiesToTemperature(double temperature) {
    double current_temp = calculateTemperature();
    // TODO: this should be a single kernel
    thrust::transform(d_velocities_x.begin(), d_velocities_x.end(), thrust::make_constant_iterator(std::sqrt(temperature / current_temp)), d_velocities_x.begin(), thrust::multiplies<double>());
    thrust::transform(d_velocities_y.begin(), d_velocities_y.end(), thrust::make_constant_iterator(std::sqrt(temperature / current_temp)), d_velocities_y.begin(), thrust::multiplies<double>());
}

void Particle::setRandomVelocities(double temperature) {
    setRandomNormal(d_velocities_x, 0.0, std::sqrt(temperature));
    setRandomNormal(d_velocities_y, 0.0, std::sqrt(temperature));
    removeMeanVelocities();
    scaleVelocitiesToTemperature(temperature);
    // thrust::fill(d_velocities.begin(), d_velocities.end(), 0.0);
}

double Particle::getDiameter(std::string which) {
    if (which == "min") {
        return 2.0 * *thrust::min_element(d_radii.begin(), d_radii.end());
    } else if (which == "max") {
        return 2.0 * *thrust::max_element(d_radii.begin(), d_radii.end());
    } else if (which == "mean") {
        return 2.0 * thrust::reduce(d_radii.begin(), d_radii.end()) / d_radii.size();
    } else {
        throw std::invalid_argument("Particle::getDiameter: which must be 'min', 'max', or 'mean', not " + which);
    }
}

void Particle::setBiDispersity(double size_ratio, double count_ratio) {
    if (size_ratio < 1.0) {
        throw std::invalid_argument("Particle::setBiDispersity: size_ratio must be > 1.0");
    }
    if (count_ratio < 0.0 || count_ratio > 1.0) {
        throw std::invalid_argument("Particle::setBiDispersity: count_ratio must be < 1.0 and > 0.0");
    }
    thrust::host_vector<double> radii(n_particles);
    long n_large = static_cast<long>(n_particles * count_ratio);
    double diam_large = size_ratio;
    double diam_small = 1.0;
    for (long i = 0; i < n_large; i++) {
        radii[i] = diam_large / 2.0;
    }
    for (long i = n_large; i < n_particles; i++) {
        radii[i] = diam_small / 2.0;
    }
    setArray("d_radii", radii);
}

double Particle::getBoxArea() {
    thrust::host_vector<double> box_size = getBoxSize();
    return thrust::reduce(box_size.begin(), box_size.end(), 1.0, thrust::multiplies<double>());
}

double Particle::getPackingFraction() {
    double box_area = getBoxArea();
    double area = getArea();
    return area / box_area;
}

double Particle::getDensity() {
    return getPackingFraction() - getOverlapFraction();
}

void Particle::scaleToPackingFraction(double packing_fraction) {
    double new_side_length = std::pow(getArea() / packing_fraction, 1.0 / N_DIM);
    double side_length = std::pow(getBoxArea(), 1.0 / N_DIM);
    scalePositions(new_side_length / side_length);
    setBoxSize(thrust::host_vector<double>(N_DIM, new_side_length));
}

double Particle::totalKineticEnergy() const {
    return thrust::reduce(d_kinetic_energy.begin(), d_kinetic_energy.end(), 0.0, thrust::plus<double>());
}

double Particle::totalPotentialEnergy() const {
    return thrust::reduce(d_potential_energy.begin(), d_potential_energy.end(), 0.0, thrust::plus<double>());
}

double Particle::totalEnergy() const {
    return totalKineticEnergy() + totalPotentialEnergy();
}

void Particle::scalePositions(double scale_factor) {
    thrust::transform(d_positions_x.begin(), d_positions_x.end(), thrust::make_constant_iterator(scale_factor), d_positions_x.begin(), thrust::multiplies<double>());
    thrust::transform(d_positions_y.begin(), d_positions_y.end(), thrust::make_constant_iterator(scale_factor), d_positions_y.begin(), thrust::multiplies<double>());
}

void Particle::updatePositions(double dt) {
    if (isnan(d_positions_x[0])) {
        std::cout << "Particle::updatePositions: NaN in positions for particle 0: pos_x=" << d_positions_x[0] << ", pos_y=" << d_positions_y[0] << std::endl;
        exit(EXIT_SUCCESS);
    }
    kernelUpdatePositions<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_last_neigh_positions_x_ptr, d_last_neigh_positions_y_ptr, d_last_cell_positions_x_ptr, d_last_cell_positions_y_ptr, d_neigh_displacements_sq_ptr, d_cell_displacements_sq_ptr, d_velocities_x_ptr, d_velocities_y_ptr, dt);
}

void Particle::updateVelocities(double dt) {
    kernelUpdateVelocities<<<dim_grid, dim_block>>>(d_velocities_x_ptr, d_velocities_y_ptr, d_forces_x_ptr, d_forces_y_ptr, d_masses_ptr, dt);
}

double Particle::getMaxSquaredNeighborDisplacement() {
    return thrust::reduce(d_neigh_displacements_sq.begin(), d_neigh_displacements_sq.end(), 0.0, thrust::maximum<double>());
}

double Particle::getMaxSquaredCellDisplacement() {
    return thrust::reduce(d_cell_displacements_sq.begin(), d_cell_displacements_sq.end(), 0.0, thrust::maximum<double>());
}

void Particle::updateNeighborList() {
    thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
    kernelUpdateNeighborList<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_last_neigh_positions_x_ptr, d_last_neigh_positions_y_ptr, d_neigh_displacements_sq_ptr, neighbor_cutoff);
    max_neighbors = thrust::reduce(d_num_neighbors.begin(), d_num_neighbors.end(), -1L, thrust::maximum<long>());
    if (max_neighbors > max_neighbors_allocated) {
        max_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_neighbors)));
        std::cout << "Particle::updateNeighborList: Resizing neighbor list to " << max_neighbors_allocated << std::endl;
        d_neighbor_list.resize(n_particles * max_neighbors_allocated);
        thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
        syncNeighborList();
        kernelUpdateNeighborList<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_last_neigh_positions_x_ptr, d_last_neigh_positions_y_ptr, d_neigh_displacements_sq_ptr, neighbor_cutoff);
    }
}

void Particle::checkNeighbors() {
    // std::cout << "Particle::checkNeighbors: Checking neighbors" << std::endl;
    (this->*checkForNeighborUpdatePtr)();
}

void Particle::checkForNeighborUpdate() {
    double tolerance = 3.0;
    double max_squared_neighbor_displacement = getMaxSquaredNeighborDisplacement();
    // std::cout << "Particle::checkForNeighborUpdate: Max squared neighbor displacement: " << tolerance * max_squared_neighbor_displacement << " vs " << neighbor_displacement << std::endl;
    if (tolerance * max_squared_neighbor_displacement > neighbor_displacement) {
        // std::cout << "Particle::checkForNeighborUpdate: Updating neighbor list" << std::endl;
        updateNeighborList();
    }
}

void Particle::checkForCellUpdate() {
    double tolerance = 3.0;
    double max_squared_cell_displacement = getMaxSquaredCellDisplacement();
    // std::cout << "Particle::checkForCellUpdate: Max squared cell displacement: " << tolerance * max_squared_cell_displacement << " vs " << cell_displacement << std::endl;
    if (tolerance * max_squared_cell_displacement > cell_displacement) {
        // std::cout << "Particle::checkForCellUpdate: Updating cell list" << std::endl;
        updateCellList();
        updateCellNeighborList();
    } else {
        double max_squared_neighbor_displacement = getMaxSquaredNeighborDisplacement();
        // std::cout << "Particle::checkForCellUpdate: Max squared neighbor displacement: " << tolerance * max_squared_neighbor_displacement << " vs " << neighbor_displacement << std::endl;
        if (tolerance * max_squared_neighbor_displacement > neighbor_displacement) {
            // std::cout << "Particle::checkForNeighborUpdate: Updating neighbor list" << std::endl;
            updateCellNeighborList();
        }
    }
}

void Particle::initializeNeighborList() {
    d_neighbor_list.resize(n_particles * max_neighbors_allocated);
    d_num_neighbors.resize(n_particles);
    thrust::fill(d_num_neighbors.begin(), d_num_neighbors.end(), 0L);
    thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
    syncNeighborList();
    updateNeighborList();
}

void Particle::setNeighborCutoff(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier) {
    this->neighbor_cutoff = neighbor_cutoff_multiplier * getDiameter("max");
    this->neighbor_displacement = neighbor_displacement_multiplier * neighbor_cutoff;
    this->max_neighbors_allocated = 4;

    thrust::host_vector<double> box_size = getBoxSize();
    std::cout << "Particle::setNeighborCutoff: Neighbor cutoff set to " << neighbor_cutoff << " and neighbor displacement set to " << neighbor_displacement << " box length: " << box_size[0] << std::endl;
}

void Particle::printNeighborList() {
    thrust::host_vector<long> neighbor_list = getArray<long>("d_neighbor_list");
    thrust::host_vector<long> num_neighbors = getArray<long>("d_num_neighbors");
    for (long i = 0; i < n_particles; i++) {
        std::cout << "Particle " << i << " has " << num_neighbors[i] << " neighbors." << std::endl;
        for (long j = 0; j < num_neighbors[i]; j++) {
            std::cout << "\t\tNeighbor " << j << " of particle " << i << " is " << neighbor_list[i * max_neighbors + j] << std::endl;
        }
    }
}

void Particle::setCellSize(double cell_size_multiplier, double cell_displacement_multiplier) {
    long min_num_cells_dim = 4;  // if there are fewer than 4 cells in one axis, the cell list probably wont work
    double trial_cell_size = cell_size_multiplier * getDiameter("max");
    thrust::host_vector<double> box_size = getBoxSize();
    n_cells_dim = static_cast<long>(std::floor(box_size[0] / trial_cell_size));
    n_cells = n_cells_dim * n_cells_dim;
    if (n_cells_dim < min_num_cells_dim) {
        throw std::runtime_error("Particle::setCellSize: fewer than " + std::to_string(min_num_cells_dim) + " cells in one dimension");
    }
    cell_size = box_size[0] / n_cells_dim;
    cell_displacement = cell_displacement_multiplier * cell_size;
    std::cout << "Particle::setCellSize: Cell size set to " << cell_size << " and cell displacement set to " << cell_displacement << std::endl;
    syncCellList();
}

void Particle::initializeCellList() {
    d_cell_index.resize(n_particles);
    d_sorted_cell_index.resize(n_particles);
    d_particle_index.resize(n_particles);
    d_cell_start.resize(n_cells + 1);

    // TODO: this should be a single kernel
    thrust::fill(d_cell_index.begin(), d_cell_index.end(), -1L);
    thrust::fill(d_sorted_cell_index.begin(), d_sorted_cell_index.end(), -1L);
    thrust::fill(d_particle_index.begin(), d_particle_index.end(), -1L);
    thrust::fill(d_cell_start.begin(), d_cell_start.end(), -1L);

    d_cell_index_ptr = thrust::raw_pointer_cast(d_cell_index.data());
    d_sorted_cell_index_ptr = thrust::raw_pointer_cast(d_sorted_cell_index.data());
    d_particle_index_ptr = thrust::raw_pointer_cast(d_particle_index.data());
    d_cell_start_ptr = thrust::raw_pointer_cast(d_cell_start.data());

    updateCellList();
}

void Particle::syncCellList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_n_cells, &n_cells, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncCellList: Error copying n_cells to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_n_cells_dim, &n_cells_dim, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncCellList: Error copying n_cells_dim to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_cell_size, &cell_size, sizeof(double));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncCellList: Error copying cell_size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::reorderParticleData() {
    // std::cout << "\tParticle::reorderParticleData: Reordering particle data" << std::endl;
    // do the initial sorting with thrust for convenience
    thrust::sort_by_key(d_sorted_cell_index.begin(), d_sorted_cell_index.end(), d_particle_index.begin());
    // reorder the data by copying into temporary arrays
    // dont bother reordering the displacements, last positions, or energies
    // now print something to see if the ptrs are correct
    kernelReorderParticleData<<<dim_grid, dim_block>>>(d_particle_index_ptr, d_positions_x_ptr, d_positions_y_ptr, d_forces_x_ptr, d_forces_y_ptr, d_velocities_x_ptr, d_velocities_y_ptr, d_masses_ptr, d_radii_ptr, d_temp_positions_x_ptr, d_temp_positions_y_ptr, d_temp_forces_x_ptr, d_temp_forces_y_ptr, d_temp_velocities_x_ptr, d_temp_velocities_y_ptr, d_temp_masses_ptr, d_temp_radii_ptr, d_last_cell_positions_x_ptr, d_last_cell_positions_y_ptr, d_cell_displacements_sq_ptr);

    // swap the pointers
    thrust::swap(d_positions_x_ptr, d_temp_positions_x_ptr);
    thrust::swap(d_positions_y_ptr, d_temp_positions_y_ptr);
    thrust::swap(d_forces_x_ptr, d_temp_forces_x_ptr);
    thrust::swap(d_forces_y_ptr, d_temp_forces_y_ptr);
    thrust::swap(d_velocities_x_ptr, d_temp_velocities_x_ptr);
    thrust::swap(d_velocities_y_ptr, d_temp_velocities_y_ptr);
    thrust::swap(d_masses_ptr, d_temp_masses_ptr);
    thrust::swap(d_radii_ptr, d_temp_radii_ptr);

    assert(d_positions_x_ptr != d_temp_positions_x_ptr);
    assert(d_positions_y_ptr != d_temp_positions_y_ptr);
    assert(d_forces_x_ptr != d_temp_forces_x_ptr);
    assert(d_forces_y_ptr != d_temp_forces_y_ptr);
    assert(d_velocities_x_ptr != d_temp_velocities_x_ptr);
    assert(d_velocities_y_ptr != d_temp_velocities_y_ptr);
    assert(d_masses_ptr != d_temp_masses_ptr);
    assert(d_radii_ptr != d_temp_radii_ptr);
}

void Particle::updateCellList() {
    d_cell_start[n_cells] = n_particles;
    kernelGetCellIndexForParticle<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_cell_index_ptr, d_sorted_cell_index_ptr, d_particle_index_ptr);

    reorderParticleData();

    // TODO: this is a kernel over cells - could probably be parallelized better
    long width_offset = 2;
    long width = n_particles / n_cells;
    kernelGetFirstParticleIndexForCell<<<dim_grid, dim_block>>>(d_sorted_cell_index_ptr, d_cell_start_ptr, width_offset, width);
}

// TODO: look into better ways to structure the grid and block sizes
void Particle::updateCellNeighborList() {
    thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
    kernelUpdateCellNeighborList<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_last_neigh_positions_x_ptr, d_last_neigh_positions_y_ptr, neighbor_cutoff, d_cell_index_ptr, d_particle_index_ptr, d_cell_start_ptr, d_neigh_displacements_sq_ptr);
    max_neighbors = thrust::reduce(d_num_neighbors.begin(), d_num_neighbors.end(), -1L, thrust::maximum<long>());
    if (max_neighbors > max_neighbors_allocated) {
        max_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_neighbors)));
        std::cout << "Particle::updateCellNeighborList: Resizing neighbor list to " << max_neighbors_allocated << std::endl;
        d_neighbor_list.resize(n_particles * max_neighbors_allocated);
        thrust::fill(d_neighbor_list.begin(), d_neighbor_list.end(), -1L);
        syncNeighborList();
        kernelUpdateCellNeighborList<<<dim_grid, dim_block>>>(d_positions_x_ptr, d_positions_y_ptr, d_last_neigh_positions_x_ptr, d_last_neigh_positions_y_ptr, neighbor_cutoff, d_cell_index_ptr, d_particle_index_ptr, d_cell_start_ptr, d_neigh_displacements_sq_ptr);
    }
}

// TODO: this should be a single kernel
void Particle::zeroForceAndPotentialEnergy() {
    kernelZeroForceAndPotentialEnergy<<<dim_grid, dim_block>>>(d_forces_x_ptr, d_forces_y_ptr, d_potential_energy_ptr);
}

double Particle::calculateTemperature() {
    calculateKineticEnergy();
    return totalKineticEnergy() * 2.0 / n_dof;
}

double Particle::getTimeUnit() {
    double average_mass = thrust::reduce(d_masses.begin(), d_masses.end()) / n_particles;
    return getDiameter("min") * std::sqrt(average_mass / getEnergyScale("c"));
}

void Particle::setMass(double mass) {
    thrust::fill(d_masses.begin(), d_masses.end(), mass);
}
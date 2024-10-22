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


#include <typeinfo>
template <typename T>
void printType(const T& obj) {
    std::cout << "Type: " << typeid(obj).name() << std::endl;
}

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
    clearNeighborVariables();
}

void Particle::initializeFromConfig(const BaseParticleConfig& config) {
    if (const auto* bidisperse_config = dynamic_cast<const BidisperseParticleConfig*>(&config)) {
        this->config = std::make_unique<BidisperseParticleConfig>(*bidisperse_config);
    } else {
        throw std::runtime_error("ERROR: Disk::initializeFromConfig: Invalid configuration type.");
    }

    this->define_unique_dependencies();

    this->setSeed(config.seed);
    this->setParticleCounts(config.n_particles, 0);
    this->setKernelDimensions(config.particle_dim_block);

    // Dynamic cast to check if config is BidisperseParticleConfig
    if (const auto* bidisperse_config = dynamic_cast<const BidisperseParticleConfig*>(&config)) {
        // Handle bidisperse-specific fields
        this->setBiDispersity(bidisperse_config->size_ratio, bidisperse_config->count_ratio);
    } else {
        throw std::runtime_error("ERROR: Disk::initializeFromConfig: Invalid configuration type.");
    }
    this->initializeBox(config.packing_fraction);

    // TODO: make this a config - position initialization config: zero, random, etc.
    this->setRandomPositions();

    this->setEnergyScale(config.e_c, "c");
    this->setExponent(config.n_c, "c");
    this->setMass(config.mass);

    this->setNeighborMethod(config.neighbor_list_update_method);
    this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);

    if (this->neighbor_list_update_method == "cell") {
        bool could_set_cell_size = this->setCellSize(config.num_particles_per_cell, config.cell_displacement_multiplier);
        if (!could_set_cell_size) {
            std::cout << "WARNING: Disk::initializeFromConfig: Could not set cell size.  Attempting to use verlet list instead." << std::endl;
            this->setNeighborMethod("verlet");
        }
        bool could_set_neighbor_size = this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
        if (!could_set_neighbor_size) {
            std::cerr << "ERROR: Disk::initializeFromConfig: Could not set neighbor size for cell list - neighbor cutoff exceeds box size.  Attempting to use all-to-all instead." << std::endl;
            this->setNeighborMethod("all");
        }
    }
    if (this->neighbor_list_update_method == "verlet") {
        bool could_set_neighbor_size = this->setNeighborSize(config.neighbor_cutoff_multiplier, config.neighbor_displacement_multiplier);
        if (!could_set_neighbor_size) {
            std::cout << "WARNING: Disk::initializeFromConfig: Could not set neighbor size.  Attempting to use all-to-all instead." << std::endl;
            this->setNeighborMethod("all");
        }
    }
    this->initNeighborList();
    this->calculateForces();  // make sure forces are calculated before the integration starts
    // may want to check that the forces are balanced
}

void Particle::setNeighborMethod(std::string method_name) {
    this->using_cell_list = false;
    this->neighbor_list_update_method = method_name;
    if (method_name == "cell") {
        std::cout << "Particle::setNeighborMethod: Setting neighbor list update method to cell" << std::endl;
        this->initNeighborListPtr = &Particle::initCellList;
        this->updateNeighborListPtr = &Particle::updateCellNeighborList;
        this->checkForNeighborUpdatePtr = &Particle::checkForCellListUpdate;
        this->using_cell_list = true;
    } else if (method_name == "verlet") {
        std::cout << "Particle::setNeighborMethod: Setting neighbor list update method to verlet" << std::endl;
        this->initNeighborListPtr = &Particle::initVerletList;
        this->updateNeighborListPtr = &Particle::updateVerletList;
        this->checkForNeighborUpdatePtr = &Particle::checkForVerletListUpdate;
    } else if (method_name == "all") {
        std::cout << "Particle::setNeighborMethod: Setting neighbor list update method to all" << std::endl;
        thrust::host_vector<double> host_box_size = box_size.getData();
        double max_diameter = getDiameter("max");
        double box_diagonal = std::sqrt(host_box_size[0] * host_box_size[0] + host_box_size[1] * host_box_size[1]);
        double neighbor_cutoff_multiplier = 2.0 * box_diagonal / max_diameter;  // set it to be twice the diagonal length so that every particle is included always (2x multiplier is extraneous but harmless)
        setNeighborSize(neighbor_cutoff_multiplier, 0.0);  // the neighbor displacement is unused here
        this->initNeighborListPtr = &Particle::initAllToAllList;
        this->updateNeighborListPtr = &Particle::updateVerletList;
        this->checkForNeighborUpdatePtr = &Particle::checkForAllToAllUpdate;
    } else {
        throw std::invalid_argument("Particle::setNeighborMethod: Invalid method name: " + method_name);
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

void Particle::setKernelDimensions(long particle_dim_block) {
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "CUDA Info: Particle::setKernelDimensions: Max threads per block: " << maxThreadsPerBlock << std::endl;
    if (particle_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: Particle::setKernelDimensions: particle_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        particle_dim_block = maxThreadsPerBlock;
    }
    this->particle_dim_block = particle_dim_block;
    // Implement some particle-specific logic to define the grid dimensions
    // Then, sync
    std::cout << "WARNING: Particle::setKernelDimensions: Not Implemented" << std::endl;
    syncKernelDimensions();
}

void Particle::syncKernelDimensions() {
    cudaError_t cuda_err;
    cuda_err = cudaMemcpyToSymbol(d_dim_block, &particle_dim_block, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying particle_dim_block to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_grid, &particle_dim_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying particle_dim_grid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cuda_err = cudaMemcpyToSymbol(d_dim_vertex_grid, &vertex_dim_grid, sizeof(long));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncKernelDimensions: Error copying vertex_dim_grid to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Particle::initDynamicVariables() {
    // Resize and fill all the device arrays to avoid any potential issues with uninitialized data
    positions.resizeAndFill(n_particles, 0.0, 0.0);
    velocities.resizeAndFill(n_particles, 0.0, 0.0);
    forces.resizeAndFill(n_particles, 0.0, 0.0);
    radii.resizeAndFill(n_particles, 0.0);
    masses.resizeAndFill(n_particles, 0.0);
    kinetic_energy.resizeAndFill(n_particles, 0.0);
    potential_energy.resizeAndFill(n_particles, 0.0);
}

void Particle::clearDynamicVariables() {
    positions.clear();
    velocities.clear();
    forces.clear();
    radii.clear();
    masses.clear();
    kinetic_energy.clear();
    potential_energy.clear();
}

void Particle::clearNeighborVariables() {
    neighbor_list.clear();
    num_neighbors.clear();
    cell_index.clear();
    particle_index.clear();
    static_particle_index.clear();
    cell_start.clear();
    last_neigh_positions.clear();
    last_cell_positions.clear();
    neigh_displacements_sq.clear();
    cell_displacements_sq.clear();
}

void Particle::define_unique_dependencies() {
    for (const auto& pair : calculation_dependencies) {
        unique_dependents.insert(pair.first);
        for (const auto& dependency : pair.second) {
            unique_dependencies.insert(dependency);
        }
    }
    reset_dependency_status();
}

void Particle::reset_dependency_status() {
    dependency_status.clear();
    for (const auto& dependency : unique_dependencies) {
        dependency_status[dependency] = false;
    }
}

void Particle::calculate_dependencies(std::string log_name) {
    for (const auto& dependency : calculation_dependencies[log_name]) {
        if (!dependency_status[dependency]) {
            calculate_dependencies(dependency);  // handle nested dependencies
            handle_calculation_for_single_dependency(dependency);
            dependency_status[dependency] = true;  // once calculated, it doesnt need to be calculated again
        }
    }
}

void Particle::handle_calculation_for_single_dependency(std::string dependency_calculation_name) {
    // logic to calculate the dependency goes here - need one for each value in unique_dependencies
    if (dependency_calculation_name == "calculate_kinetic_energy") {
        calculateKineticEnergy();
    } 
    // fill in the rest here....

    else {
        throw std::invalid_argument("Particle::handle_calculation_for_single_dependency: dependency_calculation_name not found: " + dependency_calculation_name);
    }
}

ArrayData Particle::getArrayData(const std::string& array_name) {
    ArrayData result;
    result.name = array_name;
    if (array_name == "positions") {
        result.type = DataType::Double;
        result.size = positions.size;
        result.data = std::make_pair(positions.getDataX(), positions.getDataY());
        result.index_array_name = "static_particle_index";
    } else if (array_name == "velocities") {
        result.type = DataType::Double;
        result.size = velocities.size;
        result.data = std::make_pair(velocities.getDataX(), velocities.getDataY());
        result.index_array_name = "static_particle_index";
    } else if (array_name == "forces") {
        result.type = DataType::Double;
        result.size = forces.size;
        result.data = std::make_pair(forces.getDataX(), forces.getDataY());
        result.index_array_name = "static_particle_index";
    } else if (array_name == "box_size") {
        result.type = DataType::Double;
        result.size = box_size.size;
        result.data = box_size.getData();
        result.index_array_name = "";
    } else if (array_name == "radii") {
        result.type = DataType::Double;
        result.size = radii.size;
        result.data = radii.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "masses") {
        result.type = DataType::Double;
        result.size = masses.size;
        result.data = masses.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "kinetic_energy") {
        result.type = DataType::Double;
        result.size = kinetic_energy.size;
        result.data = kinetic_energy.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "potential_energy") {
        result.type = DataType::Double;
        result.size = potential_energy.size;
        result.data = potential_energy.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "neighbor_list") {
        result.type = DataType::Long;
        result.size = neighbor_list.size;
        result.data = neighbor_list.getData();
        result.index_array_name = ""; // this is a tricky one to incorporate in the reordering process in a general way
    } else if (array_name == "num_neighbors") {
        result.type = DataType::Long;
        result.size = num_neighbors.size;
        result.data = num_neighbors.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "cell_index") {
        result.type = DataType::Long;
        result.size = cell_index.size;
        result.data = cell_index.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "particle_index") {
        result.type = DataType::Long;
        result.size = particle_index.size;
        result.data = particle_index.getData();
        result.index_array_name = "static_particle_index";
    } else if (array_name == "static_particle_index") {
        result.type = DataType::Long;
        result.size = static_particle_index.size;
        result.data = static_particle_index.getData();
        result.index_array_name = "";
    } else if (array_name == "cell_start") {
        result.type = DataType::Long;
        result.size = cell_start.size;
        result.data = cell_start.getData();
        result.index_array_name = "";
    } else {
        throw std::invalid_argument("Particle::getArrayData: array_name " + array_name + " not found");
    }
    return result;
}

void Particle::setBoxSize(const thrust::host_vector<double>& host_box_size) {  // TODO: work on this
    if (host_box_size.size() != N_DIM) {
        throw std::invalid_argument("Particle::setBoxSize: Error box_size (" + std::to_string(host_box_size.size()) + ")" + " != " + std::to_string(N_DIM) + " elements");
    }
    box_size.resize(N_DIM);
    box_size.setData(host_box_size);
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_box_size, box_size.getData().data(), sizeof(double) * N_DIM);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::setBoxSize: Error copying box size to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void Particle::syncNeighborList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_neighbors_allocated, &this->max_neighbors_allocated, sizeof(this->max_neighbors_allocated));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying max_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_neighbor_list_ptr, &neighbor_list.d_ptr, sizeof(neighbor_list.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying d_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_neighbors_ptr, &num_neighbors.d_ptr, sizeof(num_neighbors.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "Particle::syncNeighborList: Error copying d_num_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void Particle::setEnergyScale(double e, std::string which) {
    cudaError_t cuda_err;
    if (which == "c") {
        e_c = e;
        cuda_err = cudaMemcpyToSymbol(d_e_c, &e_c, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_c to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "a") {
        e_a = e;
        cuda_err = cudaMemcpyToSymbol(d_e_a, &e_a, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_a to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "b") {
        e_b = e;
        cuda_err = cudaMemcpyToSymbol(d_e_b, &e_b, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_b to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "l") {
        e_l = e;
        cuda_err = cudaMemcpyToSymbol(d_e_l, &e_l, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setEnergyScale: Error copying e_l to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
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
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "a") {
        n_a = n;
        cuda_err = cudaMemcpyToSymbol(d_n_a, &n_a, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_a to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "b") {
        n_b = n;
        cuda_err = cudaMemcpyToSymbol(d_n_b, &n_b, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_b to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
        }
    } else if (which == "l") {
        n_l = n;
        cuda_err = cudaMemcpyToSymbol(d_n_l, &n_l, sizeof(double));
        if (cuda_err != cudaSuccess) {
            std::cerr << "Particle::setExponent: Error copying n_l to device: " << cudaGetErrorString(cuda_err) << std::endl;
            exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
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
    thrust::host_vector<double> host_box_size(N_DIM, side_length);
    setBoxSize(host_box_size);
    // then rescale the box size to the desired packing fraction
    scaleToPackingFraction(packing_fraction);
}

void Particle::setRandomUniform(thrust::device_vector<double>& values, double min, double max, long seed_offset) {
    thrust::counting_iterator<long> index_sequence_begin(seed + seed_offset);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomUniform(min, max, seed));
}

void Particle::setRandomNormal(thrust::device_vector<double>& values, double mean, double stddev, long seed_offset) {
    thrust::counting_iterator<long> index_sequence_begin(seed + seed_offset);
    thrust::transform(index_sequence_begin, index_sequence_begin + values.size(), values.begin(), RandomNormal(mean, stddev, seed));
}

void Particle::setRandomPositions() {
    thrust::host_vector<double> host_box_size = box_size.getData();
    positions.fillRandomUniform(0.0, host_box_size[0], 0.0, host_box_size[1], 1, seed);
}

void Particle::removeMeanVelocities() {
    double mean_vel_x = thrust::reduce(velocities.x.d_vec.begin(), velocities.x.d_vec.end()) / velocities.x.d_vec.size();
    double mean_vel_y = thrust::reduce(velocities.y.d_vec.begin(), velocities.y.d_vec.end()) / velocities.y.d_vec.size();
    kernelRemoveMeanVelocities<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, mean_vel_x, mean_vel_y);
}

void Particle::scaleVelocitiesToTemperature(double temperature) {
    double current_temp = calculateTemperature();
    if (current_temp <= 0.0) {
        std::cout << "WARNING: Particle::scaleVelocitiesToTemperature: Current temperature is " << current_temp << ", there will be an error!" << std::endl;
    }
    double scale_factor = std::sqrt(temperature / current_temp);
    velocities.scale(scale_factor, scale_factor);
}

void Particle::setRandomVelocities(double temperature) {
    velocities.fillRandomNormal(0.0, std::sqrt(temperature), 0.0, std::sqrt(temperature), 1, seed);
    removeMeanVelocities();
    scaleVelocitiesToTemperature(temperature);
}

double Particle::getDiameter(std::string which) {
    if (which == "min") {
        return 2.0 * *thrust::min_element(radii.d_vec.begin(), radii.d_vec.end());
    } else if (which == "max") {
        return 2.0 * *thrust::max_element(radii.d_vec.begin(), radii.d_vec.end());
    } else if (which == "mean") {
        return 2.0 * thrust::reduce(radii.d_vec.begin(), radii.d_vec.end()) / radii.d_vec.size();
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
    thrust::host_vector<double> host_radii(n_particles);
    long n_large = static_cast<long>(n_particles * count_ratio);
    double diam_large = size_ratio;
    double diam_small = 1.0;
    for (long i = 0; i < n_large; i++) {
        host_radii[i] = diam_large / 2.0;
    }
    for (long i = n_large; i < n_particles; i++) {
        host_radii[i] = diam_small / 2.0;
    }
    radii.setData(host_radii);
}

double Particle::getBoxArea() {
    return thrust::reduce(box_size.d_vec.begin(), box_size.d_vec.end(), 1.0, thrust::multiplies<double>());
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
    double scale_factor = new_side_length / side_length;
    positions.scale(scale_factor, scale_factor);
    thrust::host_vector<double> host_box_size(N_DIM, new_side_length);
    setBoxSize(host_box_size);
}

double Particle::totalKineticEnergy() const {
    return thrust::reduce(kinetic_energy.d_vec.begin(), kinetic_energy.d_vec.end(), 0.0, thrust::plus<double>());
}

double Particle::totalPotentialEnergy() const {
    return thrust::reduce(potential_energy.d_vec.begin(), potential_energy.d_vec.end(), 0.0, thrust::plus<double>());
}

double Particle::totalEnergy() const {
    return totalKineticEnergy() + totalPotentialEnergy();
}

void Particle::scalePositions(double scale_factor) {
    positions.scale(scale_factor, scale_factor);
}

void Particle::updatePositions(double dt) {
    kernelUpdatePositions<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, neigh_displacements_sq.d_ptr, cell_displacements_sq.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, dt);
}

void Particle::updateVelocities(double dt) {
    kernelUpdateVelocities<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, forces.x.d_ptr, forces.y.d_ptr, masses.d_ptr, dt);

}

double Particle::getMaxSquaredNeighborDisplacement() {
    return thrust::reduce(neigh_displacements_sq.d_vec.begin(), neigh_displacements_sq.d_vec.end(), 0.0, thrust::maximum<double>());
}

double Particle::getMaxSquaredCellDisplacement() {
    return thrust::reduce(cell_displacements_sq.d_vec.begin(), cell_displacements_sq.d_vec.end(), 0.0, thrust::maximum<double>());
}

void Particle::updateVerletList() {
    neighbor_list.fill(-1L);
    kernelUpdateNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neigh_displacements_sq.d_ptr, neighbor_cutoff);
    max_neighbors = thrust::reduce(num_neighbors.d_vec.begin(), num_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
    if (max_neighbors > max_neighbors_allocated) {
        max_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_neighbors)));
        std::cout << "Particle::updateVerletList: Resizing neighbor list to " << max_neighbors_allocated << std::endl;
        neighbor_list.resize(n_particles * max_neighbors_allocated);
        neighbor_list.fill(-1L);
        syncNeighborList();
        kernelUpdateNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neigh_displacements_sq.d_ptr, neighbor_cutoff);
    }
}

void Particle::checkForAllToAllUpdate() {
    // Do nothing
}

void Particle::checkForNeighborUpdate() {
    (this->*checkForNeighborUpdatePtr)();
}

void Particle::checkForVerletListUpdate() {
    double tolerance = 3.0;
    double max_squared_neighbor_displacement = getMaxSquaredNeighborDisplacement();
    if (tolerance * max_squared_neighbor_displacement > neighbor_displacement_threshold_sq) {
        updateVerletList();
    }
}

void Particle::checkForCellListUpdate() {
    double tolerance = 3.0;
    double max_squared_cell_displacement = getMaxSquaredCellDisplacement();
    if (tolerance * max_squared_cell_displacement > cell_displacement_threshold_sq) {
        updateCellList();
        updateCellNeighborList();
    } else {
        double max_squared_neighbor_displacement = getMaxSquaredNeighborDisplacement();
        if (tolerance * max_squared_neighbor_displacement > neighbor_displacement_threshold_sq) {
            updateCellNeighborList();
        }
    }
}

void Particle::initNeighborList() {
    (this->*initNeighborListPtr)();
}

void Particle::initVerletListVariables() {
    neighbor_list.resizeAndFill(n_particles * max_neighbors_allocated, -1L);
    num_neighbors.resizeAndFill(n_particles, 0L);
    last_neigh_positions.resizeAndFill(n_particles, 0.0, 0.0);
    neigh_displacements_sq.resizeAndFill(n_particles, 0.0);
    last_cell_positions.resizeAndFill(n_particles, 0.0, 0.0);  // TODO: this is a waste of memory for non-cell list usage but would require defining a new position update kernel
    cell_displacements_sq.resizeAndFill(n_particles, 0.0);
}

void Particle::initVerletList() {
    initVerletListVariables();
    syncNeighborList();
    updateVerletList();
}

void Particle::initAllToAllListVariables() {
    this->max_neighbors_allocated = n_particles;
    initVerletListVariables();
}

void Particle::initAllToAllList() {
    initAllToAllListVariables();
    syncNeighborList();
    updateVerletList();
}

void Particle::initCellListVariables() {
    cell_index.resizeAndFill(n_particles, -1L);
    particle_index.resize(n_particles);
    static_particle_index.resize(n_particles);
    cell_start.resize(n_cells + 1);
    thrust::sequence(particle_index.d_vec.begin(), particle_index.d_vec.end());
    thrust::sequence(static_particle_index.d_vec.begin(), static_particle_index.d_vec.end());
}

void Particle::initCellList() {
    initVerletListVariables();
    syncNeighborList();
    initCellListVariables();
    updateCellList();
    updateCellNeighborList();
}

bool Particle::setNeighborSize(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier) {
    this->max_neighbors_allocated = 4;  // initial assumption, probably could be refined
    this->neighbor_cutoff = neighbor_cutoff_multiplier * getDiameter("max");
    this->neighbor_displacement_threshold_sq = std::pow(neighbor_displacement_multiplier * neighbor_cutoff, 2);
    thrust::host_vector<double> host_box_size = box_size.getData();
    double box_diagonal = std::sqrt(host_box_size[0] * host_box_size[0] + host_box_size[1] * host_box_size[1]);
    if (neighbor_cutoff >= box_diagonal) {
        std::cout << "Particle::setNeighborSize: Neighbor radius exceeds the box size" << std::endl;
        return false;
    }
    return true;
}

bool Particle::setCellSize(double num_particles_per_cell, double cell_displacement_multiplier) {
    long min_num_cells_dim = 4;  // if there are fewer than 4 cells in one axis, the cell list is spiritually defeated
    double number_density = getNumberDensity();
    double trial_cell_size = std::sqrt(num_particles_per_cell / number_density);
    double min_cell_size = 2.0 * getDiameter("max");  // somewhat arbitrary bound, probably could be refined
    thrust::host_vector<double> host_box_size = box_size.getData();
    n_cells_dim = static_cast<long>(std::floor(host_box_size[0] / trial_cell_size));
    n_cells = n_cells_dim * n_cells_dim;
    if (n_cells_dim < min_num_cells_dim) {
        std::cout << "Particle::setCellSize: fewer than " << min_num_cells_dim << " cells in one dimension" << std::endl;
        n_cells_dim = min_num_cells_dim;
        n_cells = n_cells_dim * n_cells_dim;
    }
    cell_size = host_box_size[0] / n_cells_dim;
    if (cell_size < min_cell_size) {
        std::cout << "Particle::setCellSize: cell size is less than twice the maximum diameter" << std::endl;  // 
        cell_size = min_cell_size;

        // try to make the cell again
        n_cells_dim = static_cast<long>(std::floor(host_box_size[0] / cell_size));
        n_cells = n_cells_dim * n_cells_dim;
        if (n_cells_dim < min_num_cells_dim) {
            std::cout << "Particle::setCellSize: Failed to make cell list - fewer than " << min_num_cells_dim << " cells in one dimension and cell size is less than twice the maximum particle diameter" << std::endl;
            return false;
        }
    }
    cell_displacement_threshold_sq = std::pow(cell_displacement_multiplier * cell_size, 2);
    std::cout << "Particle::setCellSize: Cell size set to " << cell_size << " and cell displacement set to " << cell_displacement_threshold_sq << " for " << n_cells << " cells" << std::endl;
    syncCellList();
    return true;
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
    thrust::sort_by_key(cell_index.d_vec.begin(), cell_index.d_vec.end(), thrust::make_zip_iterator(thrust::make_tuple(particle_index.d_vec.begin(), static_particle_index.d_vec.begin())));
    kernelReorderParticleData<<<particle_dim_grid, particle_dim_block>>>(particle_index.d_ptr, positions.x.d_ptr, positions.y.d_ptr, forces.x.d_ptr, forces.y.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, radii.d_ptr, positions.x.d_temp_ptr, positions.y.d_temp_ptr, forces.x.d_temp_ptr, forces.y.d_temp_ptr, velocities.x.d_temp_ptr, velocities.y.d_temp_ptr, masses.d_temp_ptr, radii.d_temp_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, cell_displacements_sq.d_ptr);
    positions.swap();
    forces.swap();
    velocities.swap();
    masses.swap();
    radii.swap();
}

void Particle::updateCellList() {
    cell_start.d_vec[n_cells] = n_particles;
    kernelGetCellIndexForParticle<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, cell_index.d_ptr, particle_index.d_ptr);
    reorderParticleData();
    cudaDeviceSynchronize();
    // TODO: this is a kernel over cells - could probably be parallelized better
    long width_offset = 2;
    long width = n_particles / n_cells;
    // TODO FIXXXXXX
    kernelGetFirstParticleIndexForCell<<<n_cells, particle_dim_block>>>(cell_index.d_ptr, cell_start.d_ptr, width_offset, width);
}

// TODO: look into better ways to structure the grid and block sizes
void Particle::updateCellNeighborList() {
    neighbor_list.fill(-1L);
    kernelUpdateCellNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neighbor_cutoff, cell_index.d_ptr, cell_start.d_ptr, neigh_displacements_sq.d_ptr);
    max_neighbors = thrust::reduce(num_neighbors.d_vec.begin(), num_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
    if (max_neighbors > max_neighbors_allocated) {
        max_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_neighbors)));
        std::cout << "Particle::updateCellNeighborList: Resizing neighbor list to " << max_neighbors_allocated << std::endl;
        neighbor_list.resizeAndFill(n_particles * max_neighbors_allocated, -1L);
        syncNeighborList();
        kernelUpdateCellNeighborList<<<particle_dim_grid, particle_dim_block>>>(positions.x.d_ptr, positions.y.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, neighbor_cutoff, cell_index.d_ptr, cell_start.d_ptr, neigh_displacements_sq.d_ptr);
    }
}

// TODO: this should be a single kernel
void Particle::zeroForceAndPotentialEnergy() {
    kernelZeroForceAndPotentialEnergy<<<particle_dim_grid, particle_dim_block>>>(forces.x.d_ptr, forces.y.d_ptr, potential_energy.d_ptr);
}

double Particle::calculateTemperature() {
    calculateKineticEnergy();
    return totalKineticEnergy() * 2.0 / n_dof;
}

double Particle::getTimeUnit() {
    double average_mass = thrust::reduce(masses.d_vec.begin(), masses.d_vec.end()) / n_particles;
    return getDiameter("min") * std::sqrt(average_mass / getEnergyScale("c"));
}

void Particle::setMass(double mass) {
    masses.fill(mass);
}


double Particle::getNumberDensity() {
    return n_particles / getBoxArea();
}

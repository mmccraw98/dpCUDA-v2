#include "../../include/constants.h"
#include "../../include/functors.h"
#include "../../include/particle/particle.h"
#include "../../include/particle/rigid_bumpy.h"
#include "../../include/kernels/kernels.cuh"
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

RigidBumpy::RigidBumpy() {
}

RigidBumpy::~RigidBumpy() {
}

// ----------------------------------------------------------------------
// --------------------- Overridden Methods -----------------------------
// ----------------------------------------------------------------------


void RigidBumpy::setKernelDimensions(long particle_dim_block, long vertex_dim_block) {
    int maxThreadsPerBlock;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    std::cout << "CUDA Info: Particle::setKernelDimensions: Max threads per block: " << maxThreadsPerBlock << std::endl;
    if (particle_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: Particle::setKernelDimensions: particle_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        particle_dim_block = maxThreadsPerBlock;
    }
    if (n_particles <= 0) {
        std::cout << "ERROR: Disk::setKernelDimensions: n_particles is 0.  Set n_particles before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_particles < particle_dim_block) {
        particle_dim_block = n_particles;
    }
    this->particle_dim_block = particle_dim_block;
    this->particle_dim_grid = (n_particles + particle_dim_block - 1) / particle_dim_block;

    if (vertex_dim_block > maxThreadsPerBlock) {
        std::cout << "WARNING: RigidBumpy::setKernelDimensions: vertex_dim_block exceeds maxThreadsPerBlock, adjusting to maxThreadsPerBlock" << std::endl;
        vertex_dim_block = maxThreadsPerBlock;
    }
    if (n_vertices <= 0) {
        std::cout << "ERROR: RigidBumpy::setKernelDimensions: n_vertices is 0.  Set n_vertices before setting kernel dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }
    if (n_vertices < vertex_dim_block) {
        vertex_dim_block = n_vertices;
    }
    this->vertex_dim_block = vertex_dim_block;
    this->vertex_dim_grid = (n_vertices + vertex_dim_block - 1) / vertex_dim_block;

    syncKernelDimensions();
}

void RigidBumpy::initVertexVariables() {
    vertex_positions.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_velocities.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_forces.resizeAndFill(n_vertices, 0.0, 0.0);
    vertex_torques.resizeAndFill(n_vertices, 0.0);
    vertex_particle_index.resizeAndFill(n_vertices, 0);
    vertex_masses.resizeAndFill(n_vertices, 0.0);
    vertex_potential_energy.resizeAndFill(n_vertices, 0.0);
    moments_of_inertia.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::initDynamicVariables() {
    Particle::initDynamicVariables();
    angles.resizeAndFill(n_particles, 0.0);
    angular_velocities.resizeAndFill(n_particles, 0.0);
    torques.resizeAndFill(n_particles, 0.0);
    angle_delta.resizeAndFill(n_particles, 0.0);
    delta.resizeAndFill(n_particles, 0.0, 0.0);
    particle_start_index.resizeAndFill(n_particles, 0);
    num_vertices_in_particle.resizeAndFill(n_particles, 0);
    inverse_particle_index.resizeAndFill(n_particles, 0);
}

void RigidBumpy::initGeometricVariables() {
    area.resizeAndFill(n_particles, 0.0);
}

void RigidBumpy::clearDynamicVariables() {
    Particle::clearDynamicVariables();
    vertex_positions.clear();
    vertex_velocities.clear();
    vertex_forces.clear();
    vertex_torques.clear();
    vertex_potential_energy.clear();
    angles.clear();
    angular_velocities.clear();
    torques.clear();
    vertex_particle_index.clear();
    particle_start_index.clear();
    num_vertices_in_particle.clear();
    inverse_particle_index.clear();
}

void RigidBumpy::syncVertexIndices() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_particle_start_index_ptr, &particle_start_index.d_ptr, sizeof(particle_start_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_particle_start_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertices_in_particle_ptr, &num_vertices_in_particle.d_ptr, sizeof(num_vertices_in_particle.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_num_vertices_in_particle_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_vertex_particle_index_ptr, &vertex_particle_index.d_ptr, sizeof(vertex_particle_index.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexIndices: Error copying d_vertex_particle_index_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setParticleStartIndex() {
    thrust::exclusive_scan(num_vertices_in_particle.d_vec.begin(), num_vertices_in_particle.d_vec.end(), particle_start_index.d_vec.begin());
}

void RigidBumpy::syncVertexRadius(double vertex_radius) {
    cudaMemcpyToSymbol(d_vertex_radius, &vertex_radius, sizeof(double));
}

double RigidBumpy::getVertexRadius() {
    double vertex_radius;
    cudaMemcpyFromSymbol(&vertex_radius, d_vertex_radius, sizeof(double));
    return vertex_radius;
}

long RigidBumpy::setVertexBiDispersity(long num_vertices_in_small_particle) {
    double max_particle_diam = getDiameter("max");
    double min_particle_diam = getDiameter("min");

    auto num_small_particles = thrust::count_if(
        radii.d_vec.begin(), 
        radii.d_vec.end(), 
        [=] __device__ (double rad) {
            return rad == min_particle_diam / 2.0;
        }
    );

    auto num_large_particles = n_particles - num_small_particles;
    long num_vertices_in_large_particle = static_cast<long>(num_vertices_in_small_particle * max_particle_diam / min_particle_diam);

    double vertex_angle_small = 2 * M_PI / num_vertices_in_small_particle;
    double vertex_radius = min_particle_diam / (1 + segment_length_per_vertex_diameter / std::sin(vertex_angle_small / 2)) / 2.0;
    syncVertexRadius(vertex_radius);
    setNumVertices(num_small_particles * num_vertices_in_small_particle + num_large_particles * num_vertices_in_large_particle);
    return num_vertices_in_large_particle;
}

void RigidBumpy::setDegreesOfFreedom() {
    if (rotation) {
        this->n_dof = n_particles * (N_DIM + 1);  // two translation and one rotation
    } else {
        this->n_dof = n_particles * N_DIM;  // two translation
    }
}

void RigidBumpy::scaleVelocitiesToTemperature(double temperature) {
    double current_temp = calculateTemperature();
    if (current_temp <= 0.0) {
        std::cout << "WARNING: RigidBumpy::scaleVelocitiesToTemperature: Current temperature is " << current_temp << ", there will be an error!" << std::endl;
    }
    double scale_factor = std::sqrt(temperature / current_temp);
    velocities.scale(scale_factor, scale_factor);
    angular_velocities.scale(scale_factor);
}


void RigidBumpy::setRandomVelocities(double temperature) {
    velocities.fillRandomNormal(0.0, std::sqrt(temperature), 0.0, std::sqrt(temperature), 1, seed);
    angular_velocities.fillRandomNormal(0.0, std::sqrt(temperature), 1, seed);
    removeMeanVelocities();
    scaleVelocitiesToTemperature(temperature);
}

void RigidBumpy::initializeVerticesFromDiskPacking(SwapData2D<double>& disk_positions, SwapData1D<double>& disk_radii, long num_vertices_in_small_particle, long particle_dim_block, long vertex_dim_block) {
    // set the number of particles from the disk data
    setNumParticles(disk_positions.size[0]);
    initDynamicVariables();
    initGeometricVariables();

    // set the particle positions and radii from the disk packing
    positions.copyFrom(disk_positions);
    radii.copyFrom(disk_radii);


    // define the number of vertices using the bidispersity
    long num_vertices_in_large_particle = setVertexBiDispersity(num_vertices_in_small_particle);

    setDegreesOfFreedom();

    // set the kernel dimensions
    setKernelDimensions(particle_dim_block, vertex_dim_block);

    // initialize the vertex variables
    initVertexVariables();

    double min_particle_diam = getDiameter("min");
    double max_particle_diam = getDiameter("max");

    // set the number of vertices in each particle
    kernelGetNumVerticesInParticles<<<particle_dim_grid, particle_dim_block>>>(
        radii.d_ptr, min_particle_diam, num_vertices_in_small_particle, max_particle_diam, num_vertices_in_large_particle, num_vertices_in_particle.d_ptr);

    // set the particle start index
    setParticleStartIndex();

    // initialize the vertices on the particles
    kernelInitializeVerticesOnParticles<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, radii.d_ptr, angles.d_ptr, vertex_particle_index.d_ptr, particle_start_index.d_ptr, num_vertices_in_particle.d_ptr, vertex_masses.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr);
    
    // sync the vertex indices
    syncVertexIndices();
}


ArrayData RigidBumpy::getArrayData(const std::string& array_name) {
    try {
        return Particle::getArrayData(array_name);
    } catch (std::invalid_argument& e) {
        // try the rigid bumpy specific ones
        ArrayData result;
        result.name = array_name;
        if (array_name == "vertex_positions") {
            result.type = DataType::Double;
            result.size = vertex_positions.size;
            result.data = std::make_pair(vertex_positions.getDataX(), vertex_positions.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_velocities") {
            result.type = DataType::Double;
            result.size = vertex_velocities.size;
            result.data = std::make_pair(vertex_velocities.getDataX(), vertex_velocities.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_forces") {
            result.type = DataType::Double;
            result.size = vertex_forces.size;
            result.data = std::make_pair(vertex_forces.getDataX(), vertex_forces.getDataY());
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_masses") {
            result.type = DataType::Double;
            result.size = vertex_masses.size;
            result.data = vertex_masses.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "angles") {
            result.type = DataType::Double;
            result.size = angles.size;
            result.data = angles.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "angular_velocities") {
            result.type = DataType::Double;
            result.size = angular_velocities.size;
            result.data = angular_velocities.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "torques") {
            result.type = DataType::Double;
            result.size = torques.size;
            result.data = torques.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "area") {
            result.type = DataType::Double;
            result.size = area.size;
            result.data = area.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_particle_index") {
            result.type = DataType::Long;
            result.size = vertex_particle_index.size;
            result.data = vertex_particle_index.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "particle_start_index") {
            result.type = DataType::Long;
            result.size = particle_start_index.size;
            result.data = particle_start_index.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "num_vertices_in_particle") {
            result.type = DataType::Long;
            result.size = num_vertices_in_particle.size;
            result.data = num_vertices_in_particle.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_neighbor_list") {
            result.type = DataType::Long;
            result.size = vertex_neighbor_list.size;
            result.data = vertex_neighbor_list.getData();
            result.index_array_name = "";// TODO:
        } else if (array_name == "num_vertex_neighbors") {
            result.type = DataType::Long;
            result.size = num_vertex_neighbors.size;
            result.data = num_vertex_neighbors.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "vertex_index") {
            result.type = DataType::Long;
            result.size = vertex_index.size;
            result.data = vertex_index.getData();
            result.index_array_name = "static_vertex_index";
        } else if (array_name == "static_vertex_index") {
            result.type = DataType::Long;
            result.size = static_vertex_index.size;
            result.data = static_vertex_index.getData();
            result.index_array_name = "";
        } else if (array_name == "moments_of_inertia") {
            result.type = DataType::Double;
            result.size = moments_of_inertia.size;
            result.data = moments_of_inertia.getData();
            result.index_array_name = "static_particle_index";
        } else if (array_name == "vertex_torques") {
            result.type = DataType::Double;
            result.size = vertex_torques.size;
            result.data = vertex_torques.getData();
            result.index_array_name = "static_vertex_index";

        } else {
            throw std::invalid_argument("RigidBumpy::getArrayData: array_name " + array_name + " not found");
        }
        return result;
    }
}





// need to make a scale function for the particles which can then go into the base particle class and be overridden by the rigid bumpy class so we dont have to replicate the scaleToPackingFraction function

void RigidBumpy::calculateParticleArea() {
    // kernelCalculateParticlePolygonArea<<<particle_dim_grid, particle_dim_block>>>(
    //     vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, area.d_ptr);
    kernelCalculateBumpyParticleAreaFull<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, radii.d_ptr, area.d_ptr);
}

double RigidBumpy::getParticleArea() const {
    double a = thrust::reduce(area.d_vec.begin(), area.d_vec.end(), 0.0, thrust::plus<double>());
    double box_area = getBoxArea();
    return a;
}


// calculate the particle positions first if not already done not needed for rigid bumpy
// compare polygon areas to particle areas
// calculate the contribution to the area from the vertices
void RigidBumpy::scalePositions(double scale_factor) {
    kernelScalePositions<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, scale_factor
    );
}

void RigidBumpy::syncVertexNeighborList() {
    cudaError_t cuda_err = cudaMemcpyToSymbol(d_max_vertex_neighbors_allocated, &max_vertex_neighbors_allocated, sizeof(max_vertex_neighbors_allocated));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying max_vertex_neighbors_allocated to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_vertex_neighbor_list_ptr, &vertex_neighbor_list.d_ptr, sizeof(vertex_neighbor_list.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_vertex_neighbor_list_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
    cuda_err = cudaMemcpyToSymbol(d_num_vertex_neighbors_ptr, &num_vertex_neighbors.d_ptr, sizeof(num_vertex_neighbors.d_ptr));
    if (cuda_err != cudaSuccess) {
        std::cerr << "RigidBumpy::syncVertexNeighborList: Error copying d_num_vertex_neighbors_ptr to device: " << cudaGetErrorString(cuda_err) << std::endl;
        exit(EXIT_FAILURE);  // TODO: make this a function and put it in a cuda module
    }
}

void RigidBumpy::setMass(double mass) {
    Particle::setMass(mass);
    kernelGetVertexMasses<<<particle_dim_grid, particle_dim_block>>>(
        radii.d_ptr, vertex_masses.d_ptr, masses.d_ptr);

    kernelGetMomentsOfInertia<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, vertex_masses.d_ptr, moments_of_inertia.d_ptr);

    // check if sum of vertex masses is equal to particle mass for each particle
    double total_vertex_mass = thrust::reduce(vertex_masses.d_vec.begin(), vertex_masses.d_vec.end(), 0.0, thrust::plus<double>());
    if (std::abs(total_vertex_mass / n_particles - mass) > 1e-14) {
        std::cout << "WARNING: RigidBumpy::setMass: Total vertex mass does not match particle mass" << std::endl;
    }
}

double RigidBumpy::getOverlapFraction() const {
    std::cout << "FIXME: Implement getOverlapFraction" << std::endl;
    return 0.0;
}

void RigidBumpy::calculateForces() {
    // version 1: 2 kernels, 1 vertex level and 1 particle level
    kernelCalcRigidBumpyForces1<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr
    );
    kernelCalcRigidBumpyParticleForces1<<<particle_dim_grid, particle_dim_block>>>(
        vertex_forces.x.d_ptr, vertex_forces.y.d_ptr, vertex_torques.d_ptr, vertex_potential_energy.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, potential_energy.d_ptr
    );

    // // version 2: 1 particle level kernel  -- 3 times slower than version 1
    // kernelCalcRigidBumpyForces2<<<particle_dim_grid, particle_dim_block>>>(
    //     positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, potential_energy.d_ptr
    // );
}

void RigidBumpy::updatePositions(double dt) {
    kernelUpdateRigidPositions<<<particle_dim_grid, particle_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, angles.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr, last_neigh_positions.x.d_ptr, last_neigh_positions.y.d_ptr, last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr, neigh_displacements_sq.d_ptr, cell_displacements_sq.d_ptr, velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, dt
    );
    
    // version 1: vertex level
    kernelTranslateAndRotateVertices1<<<vertex_dim_grid, vertex_dim_block>>>(
        positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr
    );

    // // version 2: particle level  -- 10% slower than version 1
    // kernelTranslateAndRotateVertices2<<<particle_dim_grid, particle_dim_block>>>(
    //     positions.x.d_ptr, positions.y.d_ptr, vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, delta.x.d_ptr, delta.y.d_ptr, angle_delta.d_ptr
    // );
}

void RigidBumpy::updateVelocities(double dt) {
    kernelUpdateRigidVelocities<<<particle_dim_grid, particle_dim_block>>>(
        velocities.x.d_ptr, velocities.y.d_ptr, angular_velocities.d_ptr, forces.x.d_ptr, forces.y.d_ptr, torques.d_ptr, masses.d_ptr, moments_of_inertia.d_ptr, dt, rotation
    );
}

void RigidBumpy::calculateKineticEnergy() {
    if (rotation) {
        kernelCalculateTranslationalAndRotationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, angular_velocities.d_ptr, moments_of_inertia.d_ptr, kinetic_energy.d_ptr);
    } else {
        kernelCalculateTranslationalKineticEnergy<<<particle_dim_grid, particle_dim_block>>>(velocities.x.d_ptr, velocities.y.d_ptr, masses.d_ptr, kinetic_energy.d_ptr);
    }
}

void RigidBumpy::calculateParticlePositions() {
    kernelCalculateParticlePositions<<<particle_dim_grid, particle_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr
    );
}

void RigidBumpy::updateVertexVerletList() {
    vertex_neighbor_list.fill(-1L);
    kernelUpdateVertexNeighborList<<<vertex_dim_grid, vertex_dim_block>>>(
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr, vertex_neighbor_cutoff, vertex_particle_neighbor_cutoff
    );
    long max_vertex_neighbors = thrust::reduce(num_vertex_neighbors.d_vec.begin(), num_vertex_neighbors.d_vec.end(), -1L, thrust::maximum<long>());
    if (max_vertex_neighbors > max_vertex_neighbors_allocated) {
        max_vertex_neighbors_allocated = std::pow(2, std::ceil(std::log2(max_vertex_neighbors)));
        vertex_neighbor_list.resize(n_vertices * max_vertex_neighbors_allocated);
        vertex_neighbor_list.fill(-1L);
        syncVertexNeighborList();
        kernelUpdateVertexNeighborList<<<vertex_dim_grid, vertex_dim_block>>>(
            vertex_positions.x.d_ptr, vertex_positions.y.d_ptr, positions.x.d_ptr, positions.y.d_ptr, vertex_neighbor_cutoff, vertex_particle_neighbor_cutoff
        );
    }
}

void RigidBumpy::updateVerletList() {
    Particle::updateVerletList();
    updateVertexVerletList();
}

void RigidBumpy::initVerletListVariables() {
    Particle::initVerletListVariables();
    vertex_neighbor_list.resizeAndFill(n_vertices * max_vertex_neighbors_allocated, -1L);
    num_vertex_neighbors.resizeAndFill(n_vertices, 0L);
    syncVertexNeighborList();
}

void RigidBumpy::initVerletList() {
    initVerletListVariables();
    syncNeighborList();
    syncVertexNeighborList();
    updateVerletList();
}

void RigidBumpy::updateCellNeighborList() {
    Particle::updateCellNeighborList();
    // TODO: i think we need to make vertex ids and other variables global in the kernels?
    updateVertexVerletList();
}

void RigidBumpy::initCellListVariables() {
    Particle::initCellListVariables();
    vertex_index.resize(n_vertices);
    static_vertex_index.resize(n_vertices);
    thrust::sequence(vertex_index.d_vec.begin(), vertex_index.d_vec.end());
    thrust::sequence(static_vertex_index.d_vec.begin(), static_vertex_index.d_vec.end());
}

double RigidBumpy::getGeometryScale() {
    double vertex_diameter = 2.0 * getVertexRadius();
    double particle_diameter = getDiameter("max");
    return vertex_diameter / particle_diameter;
}


bool RigidBumpy::setNeighborSize(double neighbor_cutoff_multiplier, double neighbor_displacement_multiplier) {
    this->max_neighbors_allocated = 4;  // initial assumption, probably could be refined
    this->max_vertex_neighbors_allocated = 4;  // initial assumption, probably could be refined
    this->neighbor_cutoff = neighbor_cutoff_multiplier * getDiameter("max");
    this->vertex_neighbor_cutoff = neighbor_cutoff_multiplier * 2.0 * getVertexRadius();
    this->vertex_particle_neighbor_cutoff = getDiameter("max");  // particles within this distance of a vertex will be checked for vertex neighbors
    this->neighbor_displacement_threshold_sq = std::pow(neighbor_displacement_multiplier * vertex_neighbor_cutoff, 2);
    thrust::host_vector<double> host_box_size = box_size.getData();
    double box_diagonal = std::sqrt(host_box_size[0] * host_box_size[0] + host_box_size[1] * host_box_size[1]);
    if (neighbor_cutoff >= box_diagonal) {
        std::cout << "Particle::setNeighborSize: Neighbor radius exceeds the box size" << std::endl;
        return false;
    }
    return true;

    // rb.vertex_neighbor_cutoff = 2.0 * vertex_diameter;  // vertices within this distance of each other are neighbors
}


void RigidBumpy::reorderParticleData() {
    // Sort particles by cell index
    thrust::sort_by_key(cell_index.d_vec.begin(), cell_index.d_vec.end(), 
        thrust::make_zip_iterator(thrust::make_tuple(
            particle_index.d_vec.begin(), 
            static_particle_index.d_vec.begin()
        ))
    );

    // Reorder particle data and create inverse mapping
    kernelReorderRigidBumpyParticleData<<<particle_dim_grid, particle_dim_block>>>(
        particle_index.d_ptr,
        positions.x.d_ptr, positions.y.d_ptr,
        forces.x.d_ptr, forces.y.d_ptr,
        velocities.x.d_ptr, velocities.y.d_ptr,
        angular_velocities.d_ptr, torques.d_ptr,
        masses.d_ptr, radii.d_ptr,
        moments_of_inertia.d_ptr,
        num_vertices_in_particle.d_ptr,
        positions.x.d_temp_ptr, positions.y.d_temp_ptr,
        forces.x.d_temp_ptr, forces.y.d_temp_ptr,
        velocities.x.d_temp_ptr, velocities.y.d_temp_ptr,
        angular_velocities.d_temp_ptr, torques.d_temp_ptr,
        masses.d_temp_ptr, radii.d_temp_ptr,
        moments_of_inertia.d_temp_ptr,
        num_vertices_in_particle.d_temp_ptr,
        last_cell_positions.x.d_ptr, last_cell_positions.y.d_ptr,
        cell_displacements_sq.d_ptr,
        inverse_particle_index.d_ptr
    );

    // Calculate new particle start indices
    thrust::exclusive_scan(
        num_vertices_in_particle.d_temp_vec.begin(),
        num_vertices_in_particle.d_temp_vec.end(),
        particle_start_index.d_temp_vec.begin()
    );

    // Reorder vertex data
    kernelReorderRigidBumpyVertexData<<<vertex_dim_grid, vertex_dim_block>>>(
        vertex_particle_index.d_ptr,
        vertex_positions.x.d_ptr, vertex_positions.y.d_ptr,
        vertex_particle_index.d_temp_ptr,
        vertex_positions.x.d_temp_ptr, vertex_positions.y.d_temp_ptr,
        particle_start_index.d_ptr,
        particle_start_index.d_temp_ptr,
        inverse_particle_index.d_ptr
    );

    // Swap all reordered data
    positions.swap();
    forces.swap();
    velocities.swap();
    angular_velocities.swap();
    torques.swap();
    masses.swap();
    radii.swap();
    moments_of_inertia.swap();
    num_vertices_in_particle.swap();
    particle_start_index.swap();
    vertex_positions.swap();
    vertex_particle_index.swap();
    first_moment.swap();
    second_moment.swap();
}
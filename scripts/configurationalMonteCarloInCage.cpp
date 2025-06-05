// place N particles down in a square box of side length L using random position coordinates
// calculate the energy of the interaction with the wall
// every M particles are paired together in a static (non-updating) neighbor list
// the interaction forces between the N//M particle groups are calculated
// in post, if the energy is less than a threshold, it is a valid configuration
// record the fraction of configurations that are valid, multiply with volume of phase space

// TODO:
// remove/reconfigure bidispersity in run config
// make rb random update faster
// reconfigure rb vertex counts in run config
// merge single and multi particle scripts
// add in pbc variant

#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/adam.h"
#include "../include/integrator/fire.h"
#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"
#include "../include/particles/factory.h"
#include "../include/particles/standard_configs.h"
#include "../include/integrator/adam.h"
#include "../include/routines/compression.h"
#include "../include/routines/initialization.h"
#include "../include/particles/factory.h"
#include "../include/io/io_utils.h"
#include "../include/io/arg_parsing.h"

int main(int argc, char** argv) {
    auto [particle, step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    // assign the run config variables
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    long num_steps = run_config["num_steps"].get<long>();
    long max_neighbors_allocated = run_config["max_neighbors_allocated"].get<long>();
    long num_voronoi_vertices = run_config["num_voronoi_vertices"].get<long>();
    bool overwrite = false;

    // load and build the neighbor lists and cage data
    Data1D<long> loaded_particle_neighbor_list = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "neighbor_list.dat").string(), max_neighbors_allocated * particle->n_particles);
    Data1D<long> loaded_num_neighbors = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "num_neighbors.dat").string(), particle->n_particles);
    Data1D<long> particle_cage_id = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "cage_system_ids.dat").string(), particle->n_particles);
    thrust::host_vector<long> particle_cage_id_copy = particle_cage_id.getData();
    long max_num_cages = *std::max_element(particle_cage_id_copy.begin(), particle_cage_id_copy.end());
    particle_cage_id_copy.clear();
    Data1D<long> cage_start_index = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "cage_start_index.dat").string(), max_num_cages + 1);
    // Data2D<double> cage_box_size = read_2d_data_from_file<double>((output_dir / "system" / "restart" / "cage_box_sizes.dat").string(), max_num_cages, 2);
    // Data2D<double> cage_center = read_2d_data_from_file<double>((output_dir / "system" / "restart" / "cage_centers.dat").string(), max_num_cages, 2);
    Data2D<double> voronoi_vertices = read_2d_data_from_file<double>((output_dir / "system" / "restart" / "voro_vertices.dat").string(), num_voronoi_vertices, 2);
    Data1D<long> voronoi_cell_size = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "voro_size.dat").string(), max_num_cages + 1);
    Data1D<long> voronoi_cell_start = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "voro_start_ids.dat").string(), max_num_cages + 1);
    particle->updateNeighborList();
    // particle->neighbor_list.copyFrom(loaded_particle_neighbor_list);
    // particle->num_neighbors.copyFrom(loaded_num_neighbors);
    // particle->syncNeighborList();
    // particle->updateReplicaNeighborList();

    // initialize the io manager
    std::string particle_type = particle->config["particle_type"].get<std::string>();
    std::vector<std::string> init_names = {"positions", "box_size", "radii"};
    if (particle_type == "RigidBumpy") {
        init_names.push_back("angles");
        init_names.push_back("num_vertices_in_particle");
    }
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e8, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, nullptr, output_dir, 10, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    while (step < num_steps) {
        // particle->setRandomCagePositions(cage_box_size, particle_cage_id, cage_start_index, cage_center, step + particle->seed);
        particle->setRandomVoronoiPositions(voronoi_vertices, voronoi_cell_size, voronoi_cell_start, particle_cage_id, cage_start_index, step + particle->seed);
        particle->updateReplicaNeighborList();
        particle->zeroForceAndPotentialEnergy();
        particle->calculateForces();
        dynamics_io_manager.log(step);
        step += 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
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
    // Data1D<long> loaded_particle_neighbor_list = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "neighbor_list.dat").string(), max_neighbors_allocated * particle->n_particles);
    // Data1D<long> loaded_num_neighbors = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "num_neighbors.dat").string(), particle->n_particles);
    Data1D<long> particle_cage_id = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "cage_system_ids.dat").string(), particle->n_particles);
    thrust::host_vector<long> particle_cage_id_copy = particle_cage_id.getData();
    long num_cages = *std::max_element(particle_cage_id_copy.begin(), particle_cage_id_copy.end()) + 1;
    particle_cage_id_copy.clear();  
    particle_cage_id.clear();
    Data1D<long> cage_start_index = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "cage_start_index.dat").string(), num_cages);
    Data1D<long> cage_size = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "cage_system_sizes.dat").string(), num_cages);
    Data2D<double> voronoi_vertices = read_2d_data_from_file<double>((output_dir / "system" / "restart" / "voro_vertices.dat").string(), num_voronoi_vertices, 2);
    Data2D<double> cage_center = read_2d_data_from_file<double>((output_dir / "system" / "restart" / "cage_centers.dat").string(), num_cages, 2);
    Data1D<double> voronoi_triangle_areas = read_1d_data_from_file<double>((output_dir / "system" / "restart" / "voro_triangle_area.dat").string(), num_voronoi_vertices);
    Data1D<long> voronoi_cell_size = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "voro_size.dat").string(), num_cages);
    Data1D<long> voronoi_cell_start = read_1d_data_from_file<long>((output_dir / "system" / "restart" / "voro_start_ids.dat").string(), num_cages);
    particle->updateNeighborList();
    particle->initReplicaNeighborList(voronoi_cell_size, cage_start_index, cage_size, max_neighbors_allocated);

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

    long max_threads = 20;
    thrust::host_vector<long> h_cage_start = cage_start_index.getData();
    std::vector<std::thread> workers;

    Data1D<double>* angles_ptr = nullptr;
    if (particle_type == "RigidBumpy") {
        if (auto* rb = dynamic_cast<RigidBumpy*>(particle.get()))
            angles_ptr = &(rb->angles);          // non-owning pointer
        else
            std::cerr << "[warn] particle_type says RigidBumpy but cast failed\n";
    }
    const bool save_angles = (angles_ptr != nullptr);

    while (step < num_steps) {
        // particle->setRandomCagePositions(cage_box_size, particle_cage_id, cage_start_index, cage_center, step + particle->seed);
        particle->setRandomVoronoiPositions(num_cages, cage_center, voronoi_triangle_areas, voronoi_vertices, voronoi_cell_size, voronoi_cell_start, particle_cage_id, cage_start_index, step + particle->seed);
        particle->updateReplicaNeighborList();
        particle->zeroForceAndPotentialEnergy();
        particle->calculateForces();

        if (step % 1000 == 0) {
            std::cout << "progress: " << static_cast<double>(step) / static_cast<double>(num_steps) << std::endl;
        }

        thrust::host_vector<double> h_x = particle->positions.x.getData();
        thrust::host_vector<double> h_y = particle->positions.y.getData();
        thrust::host_vector<double> h_pe = particle->potential_energy.getData();
        thrust::host_vector<double> h_ang;
        if (save_angles) h_ang = angles_ptr->getData();

        thrust::host_vector<double> sub_x(num_cages);
        thrust::host_vector<double> sub_y(num_cages);
        thrust::host_vector<double> sub_pe(num_cages);
        thrust::host_vector<double> sub_ang;
        if (save_angles) sub_ang.resize(num_cages);

        for (long i = 0; i < num_cages; ++i) {
            long idx = h_cage_start[i];
            sub_x[i] = h_x[idx];
            sub_y[i] = h_y[idx];
            sub_pe[i] = h_pe[idx];
            if (save_angles) sub_ang[i] = h_ang[idx];
        }

        workers.emplace_back([step,
                            dir = output_dir,
                            sx  = std::move(sub_x),
                            sy  = std::move(sub_y),
                            sp  = std::move(sub_pe),
                            save_angles = save_angles,
                            sa  = std::move(sub_ang)]() mutable
        {
            namespace fs = std::filesystem;
            fs::path step_dir = dir / "trajectories" / ("t" + std::to_string(step));
            fs::create_directories(step_dir);

            std::ofstream ofs(step_dir / "positions.dat");
            ofs.setf(std::ios::scientific); ofs << std::setprecision(17);

            const long N = static_cast<long>(sx.size());
            for (long i = 0; i < N; ++i)
                ofs << sx[i] << ' ' << sy[i] << '\n';
            
            std::ofstream ofs_pe(step_dir / "potential_energy.dat");
            ofs_pe.setf(std::ios::scientific); ofs_pe << std::setprecision(17);
            for (long i = 0; i < N; ++i)
                ofs_pe << sp[i] << '\n';

            if (save_angles) {
                std::ofstream ofs_ang(step_dir / "angles.dat");
                ofs_ang.setf(std::ios::scientific); ofs_ang << std::setprecision(17);
                for (long i = 0; i < N; ++i)
                    ofs_ang << sa[i] << '\n';
            }
        });

        if (workers.size() >= static_cast<std::size_t>(max_threads)) {
            for (auto& t : workers) t.join();
            workers.clear();
        }

        step += 1;
    }

    for (auto& t : workers) t.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
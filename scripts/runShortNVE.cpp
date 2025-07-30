#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/nve.h"
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
    long rescale_freq = run_config["rescale_freq"].get<long>();
    long num_steps = run_config["num_steps"].get<long>();
    double phi_increment = run_config["phi_increment"].get<double>();
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double temperature = run_config["temperature"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool resume = run_config["resume"].get<bool>();
    bool overwrite = true;
    

    // fix all to all list
    double cell_displacement = run_config["cell_displacement"].get<double>();
    long n_cells_per_dimension = run_config["n_cells_per_dimension"].get<long>();
    if (n_cells_per_dimension < 4) {
        std::cout << "n_cells_per_dimension must be at least 4" << std::endl;
        exit(1);
    }

    double neighbor_cutoff = run_config["neighbor_cutoff"].get<double>();
    double neighbor_displacement = run_config["neighbor_displacement"].get<double>();
    double vertex_neighbor_cutoff = run_config["vertex_neighbor_cutoff"].get<double>();
    double vertex_particle_neighbor_cutoff = run_config["vertex_particle_neighbor_cutoff"].get<double>();

    // set cell sizes
    particle->n_cells_dim = n_cells_per_dimension;
    particle->n_cells = n_cells_per_dimension * n_cells_per_dimension;
    particle->cell_size = std::sqrt(particle->getBoxArea()) / static_cast<double>(particle->n_cells_dim);
    particle->cell_displacement_threshold_sq = std::pow(cell_displacement, 2);
    particle->syncCellList();

    // set neighbor sizes
    particle->max_neighbors_allocated = 4;  // arbitrary initial guess
    particle->neighbor_cutoff = neighbor_cutoff;
    particle->neighbor_displacement_threshold_sq = std::pow(neighbor_displacement, 2);
    particle->vertex_neighbor_cutoff = vertex_neighbor_cutoff;
    particle->vertex_particle_neighbor_cutoff = vertex_particle_neighbor_cutoff;

    particle->initNeighborList();
    particle->calculateForces();
    double force_balance = particle->getForceBalance();
    if (force_balance / particle->n_particles / particle->e_c > 1e-14) {
        std::cout << "WARNING: Particle::setupNeighbors: Force balance is " << force_balance << ", there will be an error!" << std::endl;
    }

    // bool could_set_neighbor_size = this->setNeighborSize(neighbor_list_config.at("neighbor_cutoff_multiplier").get<double>(), neighbor_list_config.at("neighbor_displacement_multiplier").get<double>());



    if (resume) {
        std::cout << "Resuming from: " << output_dir << std::endl;
        output_dir = run_config["input_dir"].get<std::filesystem::path>();
    }

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / particle->getTimeUnit());
    NVE nve(*particle, nve_config_dict);
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    
    if (!resume) {
        std::cout << "Starting compression and NVT relaxation run" << std::endl;
        particle->setRandomVelocities(temperature);
        double phi = particle->getPackingFraction();
        particle->scaleToPackingFractionFull(phi + phi_increment);
        phi = particle->getPackingFraction();
        particle->config["packing_fraction"] = phi;
        std::vector<ConfigDict> nvt_logger_configs = {console_config};
        IOManager nvt_io_manager(nvt_logger_configs, *particle, &nve, output_dir, 20, overwrite);
        long relaxation_step = 0;
        while (relaxation_step < num_steps / 10) {
            nve.step();
            nvt_io_manager.log(relaxation_step);
            relaxation_step++;
            if (relaxation_step % rescale_freq == 0) {
                particle->removeMeanVelocities();
                particle->scaleVelocitiesToTemperature(temperature);
            }
        }
        std::cout << "Starting NVE dynamics run" << std::endl;
    } else {
        std::cout << "Resuming NVE dynamics run" << std::endl;
    }

    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_dir, 20, !resume);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");
    auto start_time = std::chrono::high_resolution_clock::now();
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }
    dynamics_io_manager.log(step, true);
    std::cout << "Done with path: " << output_dir << std::endl;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
    return 0;
}
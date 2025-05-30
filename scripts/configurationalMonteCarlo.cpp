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
    long replica_system_size = run_config["replica_system_size"].get<long>();
    bool use_pbc = run_config["use_pbc"].get<bool>();
    double packing_fraction = run_config["packing_fraction"].get<double>();
    bool overwrite = true;

    // set box size to give a maximum packing fraction of 60% for each replica system
    double diam = particle->getDiameter("max");
    double box_length = std::sqrt(replica_system_size * PI * std::pow(diam / 2, 2) / packing_fraction);
    // std::cout << particle->box_size.d_ptr[0] << std::endl;
    std::cout << box_length << std::endl;
    thrust::host_vector<double> box_size(2);
    box_size[0] = box_length;
    box_size[1] = box_length;
    particle->setBoxSize(box_size);
    particle->initReplicaNeighborList(replica_system_size);

    std::string particle_type = particle->config["particle_type"].get<std::string>();
    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 100, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, nullptr, output_dir, 10, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    while (step < num_steps) {
        particle->setRandomPositions(step + particle->seed);
        if (replica_system_size > 1) {
            particle->updateReplicaNeighborList();
        }
        particle->zeroForceAndPotentialEnergy();
        if (!use_pbc) {
            particle->calculateWallForces();
        }
        particle->calculateForces();
        dynamics_io_manager.log(step);
        step += 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/particle/particle_factory.h"
#include "../include/io/utils.h"
#include "../include/io/console_log.h"
#include "../include/io/energy_log.h"
#include "../include/io/io_manager.h"
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

#include <nlohmann/json.hpp>

int main() {

    // TODO: add cell linked list (only check neighbors in current cell and neighboring cells (maybe can define particles as being on the boundary of a cell in parallel and can then fetch the boundary particles))

    // TODO: make a runparams object (base class that can be serialized / deserialized)
    // the run config would be a concrete subclass of an abstract class
    // the run config would be defined for each script
    // then, arg parsing could be performed to fill in the values that are missing (either from cli or file)

    // script flow:
    // arguments (cli or file or both)
    // V
    // arg parser -> (run config, particle config, integrator config, log config)
    // V
    // particle(particle config)
    // integrator(integrator config)
    // log(log config)
    // V
    // run simulation

    // TODO: make the file io (input works with particle factory) (may need to make a particle method to construct values that are missing (some values can be derived from others))
    // TODO: make an arg parsing system with defaults and overrides
        // 1: from cli
        // 2: from file

    // to make for the first time:
    // seed, particle counts, vertex counts, kernel dimensions, bidispersity values (2), packing fraction, energy scales, 1, neighbor cutoff

    // to make from a file:
    // seed, particle counts, vertex counts, kernel dimensions, radii, masses, positions, velocities, energy values
    // load everything from the parameters that is found
    // calculate values that are missing and derivable
    // if any non-derivable values are missing, throw an error
    // once loaded, handle the cmd-line arguments

    // constructing the object

    // set seed to -1 to use the current time
    // TODO: make a config from a file
    
    double neighbor_cutoff_multiplier = 2.5;  // particles within this multiple of the maximum particle diameter will be considered neighbors
    double neighbor_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the neighbor cutoff, the neighbor list will be updated
    double cell_size_multiplier = 5.0;  // cells will be roughly this multiple of the maximum particle diameter
    double cell_displacement_multiplier = 0.5;  // if the maximum displacement of a particle exceeds this multiple of the cell size, the cell list will be updated
    BidisperseDiskConfig config(0, 512, 1.0, 1.0, 2.0, 0.5, neighbor_cutoff_multiplier, neighbor_displacement_multiplier, cell_size_multiplier, cell_displacement_multiplier, "cell", 256, 1.4, 0.5);
    auto particle = create_particle(config);

    // TODO: check if switching to SoA gives a performance boost

    // TODO: move the cuda check to functors

    // TODO: reorganize the particle code

    // TODO: intelligently set the cell sizing based on particle diameter and expected cell occupancy (density is global, so can only specify the number of particles per cell)
    // TODO: make the number of cells a power of 2 - why is this important?  ask chatgpt

    // TODO: fix remove mean velocities
    // TODO: around or above 500k particles, there is an out of bounds error somewhere - probably the array needs to laid out differently
    // TODO: add unit tests and integration tests (nve energy conservation etc.)

    // TODO: improve particle configs - shouldnt have to replicate all the values for each particle

    // TODO: improve the management of the particle arrays to facilitate future work with io and orchestration

    // TODO: fix the get/set array methods

    // TODO: improve the io and orchestration stuff

    // TODO: the calc-prereq values should be handled by the orchestrator 

    // TODO: precalculate the filenames in all the logs when relevant

    // TODO: add better comments and formatting to the code

    // TODO: add high performance animation support in dptools

    // TODO: in dptools add a step attribute to configuration so can do config.step to get the current step

    // TODO: in dptools rename trajectory to trajectories to match folder name

    
    particle->setRandomVelocities(1e-3);

    // make the integrator
    double dt_dimless = 1e-2;
    NVEConfig nve_config(dt_dimless * particle->getTimeUnit());
    std::cout << "dt: " << nve_config.dt << std::endl;
    NVE nve(*particle, nve_config);

    long num_steps = 1e5;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e3;
    long min_state_save_decade = 1e1;

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        // config_from_names_lin({"step", "KE", "PE", "TE", "T"}, num_steps, num_energy_saves, "energy"),  // saves the energy data to the energy file
        config_from_names_lin_everyN({"step", "KE/N", "PE/N", "TE/N", "T"}, 1e4, "console"),  // logs to the console
        // config_from_names_lin_everyN({"step", }, 1e4, "console"),  // logs to the console
        // config_from_names_lin({"positions", "velocities", "cell_index", "sorted_cell_index", "particle_index", "cell_start", "num_neighbors", "neighbor_list"}, num_steps, num_state_saves, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names_lin({"positions_x", "positions_y", "velocities_x", "velocities_y", "forces_x", "forces_y", "potential_energy", "kinetic_energy"}, num_steps, num_state_saves, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        // config_from_names_log({"positions", "velocities"}, num_steps, num_state_saves, min_state_save_decade, "state"),  // TODO: connect this to the derivable (and underivable) quantities in the particle
        config_from_names({"radii", "masses", "positions_x", "positions_y", "velocities_x", "velocities_y", "box_size"}, "init")  // TODO: connect this to the derivable (and underivable) quantities in the particle
    };

    // TODO: do something if there is data in the folder already

    IOManager io_manager(log_group_configs, *particle, &nve, "/home/mmccraw/dev/data/24-10-14/debugging-dpcuda2", true);
    io_manager.write_params();

    // add a start time and an end time to the io manager, should be added to the config file - the end time will be used to determine if the program finished (if empty,it didnt finish)

    // io_manager.log(0);


    // TODO:
    // make state loading function (static method to load the particle from the file)
        // from file(file, )
    // make restart file and init file
    // make argument parser for defaults and overrides
    // add docstrings
    // may need to add an integrator get state method to allow integrator to save its variables

    long step = 0;

    // start the timer
    auto start = std::chrono::high_resolution_clock::now();

    while (step < num_steps) {
        nve.step();
        // std::cout << "Step: " << step << std::endl;
        io_manager.log(step);
        step++;
    }

    // stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time taken: " << duration << " milliseconds" << std::endl;

    return 0;
}

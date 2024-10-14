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

    // TODO: make a runparams object (base class that can be serialized / deserialized)

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
    BidisperseDiskConfig config(0, 1024, 1.0, 1.0, 2.0, 0.5, 1.5, 256, 1.4, 0.5);  // TODO: make a config from a file
    auto particle = create_particle(config);

    // TODO: fix remove mean velocities

    particle->setRandomVelocities(1e-3);

    // make the integrator
    double dt_dimless = 1e-2;
    NVEConfig nve_config(dt_dimless * particle->getTimeUnit());
    NVE nve(*particle, nve_config);

    long num_steps = 1e5;
    long num_energy_saves = 1e2;
    long num_state_saves = 1e1;
    long min_state_save_decade = 1e1;

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin({"step", "TE", "KE", "PE", "T"}, num_steps, num_energy_saves, "energy"),  // saves the energy data to the energy file
        config_from_names_lin_everyN({"step", "T", "TE/N"}, 1e4, "console"),  // logs to the console
        config_from_names_log({"positions", "velocities"}, num_steps, num_state_saves, min_state_save_decade, "state")
    };

    IOManager io_manager(log_group_configs, *particle, &nve, "/home/mmccraw/dev/dpCUDA/old", false);
    io_manager.write_params();

    // io_manager.log(0);


    // TODO:
    // make state loading function (static method to load the particle from the file)
        // from file(file, )
    // make restart file and init file
    // make argument parser for defaults and overrides
    // add docstrings
    // may need to add an integrator get state method to allow integrator to save its variables

    long step = 0;

    while (step < num_steps) {
        nve.step();
        io_manager.log(step);
        step++;
    }

    return 0;
}

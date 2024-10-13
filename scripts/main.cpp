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

    // TODO: make a particle factory
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

    BidisperseDiskConfig config(0, 1024, 1.0, 1.0, 2.0, 0.5, 1.5, 256, 1.4, 0.5);

    write_json_to_file("/home/mmccraw/dev/dpCUDA/old/system/config.json", config.to_json());

    auto particle = create_particle(config);

    // TODO: fix remove mean velocities


    particle->setRandomVelocities(1e-3);

    // make the integrator
    double dt = 1e-2 * particle->getTimeUnit();

    NVE nve(*particle, dt);

    // Make the io manager
    std::vector<LogGroupConfig> log_group_configs = {
        config_from_names_lin({"step", "TE", "KE", "PE", "T"}, 1e4, 100, "energy"),
        config_from_names_lin_everyN({"step", "T", "TE/N"}, 1e2, "console")
    };

    IOManager io_manager(*particle, nve, log_group_configs, "/home/mmccraw/dev/dpCUDA/old", false);

    io_manager.write_log_configs("/home/mmccraw/dev/dpCUDA/old");

    io_manager.log(0);


    // TODO:
    // make state log
    // make state loading function (static method to load the particle from the file)
        // from file(file, )
    // make restart file and init file
    // make argument parser for defaults and overrides
    // add docstrings
    // may need to add an integrator get state method to allow integrator to save its variables

    // long step = 0;

    // while (step < 1e4) {
    //     nve.step();

    //     // WRAP THIS IN SOME FUNCTION:

    //     //////////////////////////////

    //     step++;
    // }

    return 0;
}

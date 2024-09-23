#include "../include/constants.h"
#include "../include/particle/particle.h"
#include "../include/particle/disk.h"
#include "../include/integrator/nve.h"
#include "../include/io/orchestrator.h"
#include "../include/io/utils.h"
#include "../include/io/console_log.h"
#include "../include/io/energy_log.h"
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

int main() {

    // TODO: make a particle factory
    // TODO: make the file io (input works with particle factory) (may need to make a particle method to construct values that are missing (some values can be derived from others))


    // constructing the object

    Disk particle;

    particle.setSeed(0);

    // set/sync number of vertices/particles, define the array sizes
    particle.setParticleCounts(1024, 0);

    // set/sync kernel dimensions
    particle.setKernelDimensions(256);  // TODO: not sure how to best motivate this

    // define the particle sizes, initialize the box to a set packing fraction, and set random positions
    particle.setBiDispersity(1.4, 0.5);  // TODO: define scaling to determine geometry units (min, max, or mean)
    particle.initializeBox(0.5);
    particle.setRandomPositions();
    // define geometry when relevant (i.e. initialize vertex configurations, calculate shape parameters, etc.)

    // set/sync energies
    particle.setEnergyScale(1.0, "c");
    particle.setExponent(2.0, "c");
    particle.setMass(1.0);
    // TODO: set timestep
    
    particle.setRandomVelocities(1e-3);

    // define the neighbor cutoff size
    particle.setNeighborCutoff(1.5);  // 1.5 * min_diameter

    // update the neighbor list
    particle.updateNeighborList();

    // make the integrator
    NVE nve(particle, 0.001);


    // Make the orchestrator
    Orchestrator orchestrator(particle);
    // Orchestrator orchestrator(particle, &nve);  // example of passing the integrator
    
    // Make the energy log
    LogGroupConfig energy_log_config;  // TODO: construct the entire log using a from log names, save freq, save type, etc....
    energy_log_config.log_names = {"step", "TE", "KE", "PE", "T"};
    energy_log_config.save_style = "lin";
    energy_log_config.save_freq = 100;
    EnergyLog energy_log(energy_log_config, orchestrator, "/home/mmccraw/dev/dpCUDA/old/energy.csv");  // TODO: make filename config from the io manager

    // Make the console log
    LogGroupConfig console_log_config;  // TODO: construct the entire log using a from log names, save freq, save type, etc....
    console_log_config.log_names = {"step", "T", "TE/N"};
    console_log_config.save_style = "lin";
    console_log_config.save_freq = 1000;
    ConsoleLog console_log(console_log_config, orchestrator);


    std::vector<BaseLogGroup*> log_groups;
    log_groups.push_back(&energy_log);
    log_groups.push_back(&console_log);


    // TODO:
    // make functions to build the log groups
    // make io manager
    // make state log
    // make state loading function (separate so can create particle object without needing the particle object to be defined)
    // make restart file and init file
    // make argument parser for defaults and overrides

    long step = 0;


    while (step < 1e4) {
        nve.step();

        // WRAP THIS IN SOME FUNCTION:
        bool log_required = false;
        for (BaseLogGroup* log_group : log_groups) {
            log_group->update_log_status(step);
            if (log_group->should_log) {
                log_required = true;
            }
        }

        if (log_required) {
            orchestrator.init_pre_req_calculation_status();
            for (BaseLogGroup* log_group : log_groups) {
                if (log_group->should_log) {
                    log_group->log(step);
                }
            }
        }
        //////////////////////////////

        step++;
    }

    return 0;
}

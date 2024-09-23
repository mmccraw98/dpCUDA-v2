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
    EnergyLog energy_log = EnergyLog::from_names_lin(orchestrator, "/home/mmccraw/dev/dpCUDA/old/energy.csv", {"step", "TE", "KE", "PE", "T"}, 1e4, 100);


    // Make the console log
    ConsoleLog console_log = ConsoleLog::from_names_lin(orchestrator, {"step", "T", "TE/N"}, 1e4, 10);


    std::vector<BaseLogGroup*> log_groups;
    log_groups.push_back(&energy_log);
    log_groups.push_back(&console_log);


    // TODO:
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

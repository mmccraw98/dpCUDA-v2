#include "../include/particles/disk/disk.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

#include "../include/particles/standard_configs.h"

#include "../include/integrator/adam.h"

#include "../include/routines/compression.h"

#include "../include/routines/initialization.h"

#include "../include/routines/minimization.h"

#include "../include/io/io_utils.h"

int main() {

    // load from restart, init, or frame
    // return frame from each case (restart = last frame, init = first frame, frame = frame)
    // for each case, load all the fundamental values
    // for frame, if the fundamental values are not found in the frame, get them from the restart file and give a warning

    std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/disk/test-2";
    std::string input_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/disk/test";


    Disk disk;
    disk.load(input_path, "restart");

    // IF RESUMING A RUN, DO NOT OVERWRITE THE INIT FILE

    long num_steps = 1e5;
    long save_every_N_steps = 1e3;
    double dt_dimless = 1e-2;
    double temperature = 1e-3;
    bool overwrite = true;

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless * disk.getTimeUnit());
    NVE nve(disk, nve_config_dict);

    std::vector<std::string> init_names = disk.getFundamentalValues();
    std::vector<std::string> state_names = {"positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j", "cell_index", "cell_start"};
    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
        config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
        config_from_names_lin_everyN(init_names, save_every_N_steps, "restart")
    };

    IOManager dynamics_io_manager(log_group_configs, disk, &nve, output_path, 1, overwrite);
    dynamics_io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }

    return 0;
}
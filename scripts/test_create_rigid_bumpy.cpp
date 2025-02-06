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

int main() {

    ConfigDict config = get_standard_rigid_bumpy_config(32, 0.2);
    bool minimize = true;
    RigidBumpy rigid_bumpy;
    rigid_bumpy.initializeFromConfig(config, minimize);

    long num_steps = 1e6;
    long save_every_N_steps = 1e3;
    double dt_dimless = 1e-2;
    double temperature = 1e-3;
    bool overwrite = true;
    std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/rb/test";
    
    rigid_bumpy.setRandomVelocities(temperature);

    std::cout << "TIME UNIT: " << rigid_bumpy.getTimeUnit() << std::endl;

    ConfigDict nve_config_dict = get_nve_config_dict(dt_dimless / rigid_bumpy.getTimeUnit());
    NVE nve(rigid_bumpy, nve_config_dict);

    std::vector<std::string> init_names = rigid_bumpy.getFundamentalValues();
    std::vector<std::string> state_names = {"vertex_forces", "vertex_positions", "positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j", "cell_index", "cell_start", "angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_separation_angle"};
    std::vector<ConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
        config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
        config_from_names_lin_everyN(init_names, save_every_N_steps, "restart")
    };

    IOManager dynamics_io_manager(log_group_configs, rigid_bumpy, &nve, output_path, 1, overwrite);
    dynamics_io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }
    return 0;
}
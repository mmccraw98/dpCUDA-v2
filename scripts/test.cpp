#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/particles/base/config.h"

#include "../include/integrator/nve.h"

#include "../include/io/io_manager.h"
#include "../include/io/base_log_groups.h"

#include "../include/particles/factory.h"

#include "../include/particles/standard_configs.h"

#include "../include/integrator/adam.h"

#include "../include/routines/compression.h"

int main() {

    // // std::string input_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/disk/start/";
    // // std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/disk/end/";

    // std::string input_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/rigid-bumpy/start/";
    // std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/rigid-bumpy/end/";

    // long trajectory_frame = -1;
    // std::filesystem::path input_path_obj(input_path);
    // auto [particle, frameNumber, sysPath, trajPath] = loadParticleFromRoot(input_path_obj, trajectory_frame);


    // double dt_dimless = 1e-2;
    // NVEConfigDict nve_config_dict;
    // nve_config_dict["dt"] = dt_dimless * particle->getTimeUnit();
    // NVE nve(*particle, nve_config_dict);

    // long num_steps = 1e5;
    // long save_every_N_steps = 1e2;
    // bool overwrite = true;

    // std::vector<std::string> init_names = {"radii", "masses", "positions", "velocities", "forces", "box_size"};
    // std::vector<std::string> state_names = {"positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j"};
    // std::string particle_type = particle->getConfig()["type_name"];
    // if (particle_type == "RigidBumpy") {
    //     std::vector<std::string> additional_log_names = {"angles", "vertex_positions", "angular_velocities", "torques", "vertex_forces", "static_vertex_index", "vertex_index", "particle_start_index", "vertex_particle_index", "num_vertices_in_particle"};
    //     for (const auto& name : additional_log_names) {
    //         init_names.push_back(name);
    //         state_names.push_back(name);
    //     }
    // }
    // std::vector<LogGroupConfigDict> log_group_configs = {
    //     config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
    //     config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
    //     config_from_names(init_names, "init"),
    //     config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
    // };

    // IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_path, 1, overwrite);
    // dynamics_io_manager.write_params();

    // long step = 0;
    // while (step < num_steps) {
    //     dynamics_io_manager.log(step);  // moved this so we can see the first step without error
    //     nve.step();
    //     step++;
    // }






    double temperature = 1e-4;
    double dt_dimless = 1e-3;

    long n_particles = 32;
    double packing_fraction = 0.2;
    std::string particle_type = "RigidBumpy";
    // std::string particle_type = "Disk";
    long save_every_N_steps = 1e2;
    long num_compression_steps = 1e4;
    bool overwrite = true;
    // std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/disk/start/";
    std::string output_path = "/home/mmccraw/dev/data/25-02-01/loading-debugging/rigid-bumpy/start/";

    ConfigDict particle_config;
    if (particle_type == "Disk") {
        particle_config = get_standard_disk_config(n_particles, packing_fraction);
    } else if (particle_type == "RigidBumpy") {
        particle_config = get_standard_rigid_bumpy_config(n_particles, packing_fraction);
    } else {
        throw std::runtime_error("Invalid particle type: " + particle_type);
    }

    auto particle = createParticle(particle_config);
    
    long num_steps = 1e5;

    NVEConfigDict nve_config_dict;
    nve_config_dict["dt"] = dt_dimless * particle->getTimeUnit();
    NVE nve(*particle, nve_config_dict);

    std::vector<std::string> init_names = {"radii", "masses", "positions", "velocities", "forces", "box_size"};
    std::vector<std::string> state_names = {"positions", "velocities", "box_size", "forces", "static_particle_index", "particle_index", "force_pairs", "distance_pairs", "pair_ids", "overlap_pairs", "radsum_pairs", "pos_pairs_i", "pos_pairs_j", "cell_index", "cell_start"};
    particle_type = particle->getConfig()["type_name"];
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> additional_log_names = {"angles", "vertex_positions", "angular_velocities", "torques", "vertex_forces", "static_vertex_index", "vertex_index", "particle_start_index", "vertex_particle_index", "num_vertices_in_particle"};
        for (const auto& name : additional_log_names) {
            init_names.push_back(name);
            state_names.push_back(name);
        }
    }
    std::vector<LogGroupConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "KE/N", "TE/N", "T", "phi"}, 1e3, "console"),
        config_from_names_lin_everyN({"step", "PE", "KE", "TE", "T", "phi"}, save_every_N_steps, "energy"),
        config_from_names(init_names, "init"),
        config_from_names_lin_everyN(state_names, save_every_N_steps, "state"),
    };

    particle->setRandomVelocities(temperature);

    IOManager dynamics_io_manager(log_group_configs, *particle, &nve, output_path, 1, overwrite);
    dynamics_io_manager.write_params();

    long step = 0;
    while (step < num_steps) {
        nve.step();
        dynamics_io_manager.log(step);
        step++;
    }

    // TODO:
    // fix the loading issues
        // run a disk simulation
        // run from end of disk simulation
        // trajectories should be continuous

    // calibrate the temperature using the geometric factor?
    
    // measure the dependence of the energy fluctuations on the timestep and temperature
        // find good dt for bumpy and disk
    // find good neighbor cutoff values

    // add load, save, and reorder functions to the Array class and remove ArrayData
    // base the io stuff off of the Array class

    // put the area calculation in the init geometric variables function

    // fix rigid bumpy energy conservation?
    
    // add an argument to the init script to minimize the configuration (add to disks)
    // split up the init script into separate sections

    // fix the loading for rigid bumpy particles
    // simplify the code
    // add pair angles to the pair logging in rigid bumpy code
    // make a restart log group (like a state log and an init log combined) (saves frame number too)
    // fix the multithreading issue in the io_manager

    // make scripts - run nve, compress, run nvt
    // pass arguments into scripts
    // handle data outputs from scripts (compress / decompress)

    // angular stiffness
        // angle steps in c(theta, t)
    
    // isostaticity of fricitonal particles has something to do with the fragility or something
    // Fragility will depend on proportion of types of contact pairs
    // stay in between translation and rotation glass transition
        // does the range of possible jamming densities (for rb) and the difference in densities between trans and rot glass transitions
        // depend on the same thing?  i.e. there seems to be a broad range of densities where the system is arrested both statically (jammed) 
        // and dynamically (glassy).  at the lower density, the system is probably isostatic in the sense of vertex contacts (mostly 1-1 contacts)
        // but at the higher density, the system is probably hyperstatic (mostly 2-1 or 2-2 contacts).  this should relate to the fragility
        // and the structure of the pair correlation function
        // further, in this view, the low end of the density range for rbs, the rbs should act as disks with the outer radius of the rbs

    // lifetimes of types of contact pairs
        // time correlation of the contact type

    // meeting next week mpu


    // predict the steepness of the interactions from the types of contact pairs (sum over contact types i ( effective_force_i * proportion_i ))


    return 0;
}
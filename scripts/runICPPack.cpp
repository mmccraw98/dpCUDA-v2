#include "../include/particles/disk/disk.h"
#include "../include/particles/rigid_bumpy/rigid_bumpy.h"
#include "../include/integrator/damped_nve.h"
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

// NOTE:
// if the initial packing fraction is too low and a cell-list is used, the cells may be too small to contain any particles
// and the cell-list may fail once compressed to higher densities

int main(int argc, char** argv) {
    auto [particle, compression_step, console_config, energy_config, state_config, run_config] = load_configs(argc, argv);

    std::string particle_type = particle->getConfig().at("particle_type").get<std::string>();

    // assign the run config variables
    long num_compression_steps = run_config["num_compression_steps"].get<long>();
    double dt_dimless = run_config["dt_dimless"].get<double>();
    double damping_coefficient = run_config["damping_coefficient"].get<double>();
    double avg_ke_target = run_config["avg_ke_target"].get<double>();
    double avg_ke_thresh = run_config["avg_ke_thresh"].get<double>();
    double avg_pe_target = run_config["avg_pe_target"].get<double>();
    double avg_pe_threshold = run_config["avg_pe_threshold"].get<double>();
    double pe_step = run_config["pe_step"].get<double>();
    std::filesystem::path output_dir = run_config["output_dir"].get<std::filesystem::path>();
    bool overwrite = true;

    long n_particles = particle->n_particles;

    ConfigDict damped_nve_config = get_damped_nve_config_dict(dt_dimless * particle->getTimeUnit() * particle->getGeometryScale(), damping_coefficient);
    DampedNVE damped_nve(*particle, damped_nve_config);

    std::vector<std::string> init_names = particle->getFundamentalValues();
    std::vector<std::string> pair_names = {"force_pairs", "distance_pairs", "overlap_pairs", "radsum_pairs", "pair_separation_angle", "pair_ids", "potential_pairs", "contact_counts"};
    if (particle_type == "RigidBumpy") {
        std::vector<std::string> rb_pair_names = {"angle_pairs_i", "angle_pairs_j", "this_vertex_contact_counts", "pair_friction_coefficient"};
        pair_names.insert(pair_names.end(), rb_pair_names.begin(), rb_pair_names.end());
    }
    init_names.insert(init_names.end(), pair_names.begin(), pair_names.end());
    std::vector<ConfigDict> log_group_configs = {
        console_config, energy_config, state_config, config_from_names_lin_everyN(init_names, 1e4, "restart")
    };
    IOManager dynamics_io_manager(log_group_configs, *particle, &damped_nve, output_dir, 20, overwrite);
    dynamics_io_manager.write_params();
    run_config.save(output_dir / "system" / "run_config.json");

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    double D_h = -1.0, D_l = -1.0, D = 1.0, scale_factor;
    double e_c_particle = particle->getEnergyScale("c") / particle->getGeometryScale();
    double e_c = particle->getEnergyScale("c");
    avg_pe_target *= e_c;
    avg_pe_threshold *= e_c;

    // compression loop
    while (compression_step < num_compression_steps) {
        damped_nve.step();
        dynamics_io_manager.log(compression_step);
        particle->calculateKineticEnergy();
        double avg_ke = particle->totalKineticEnergy() / n_particles;
        double avg_pe = particle->totalPotentialEnergy() / n_particles;

        // IN COARSE PHASE: pe < pe thresh, compress
        if (avg_pe < avg_pe_threshold && D_h == -1.0) {  // compress with big steps
            double old_D = D;
            D += sqrt(pe_step / e_c_particle);
            scale_factor = old_D / D;
            double phi = particle->getPackingFraction();
            particle->scaleToPackingFractionFull(phi / (scale_factor * scale_factor));
            particle->removeMeanVelocities();
        }
        // IN FINE PHASE: pe < pe target * 100, compress with bisection
        else if (avg_pe < avg_pe_target * 100 && D_h != -1.0) {  // compress with bisection
            double old_D = D;
            D_l = D;
            D = (D_h + D_l) / 2.0;
            scale_factor = old_D / D;
            double phi = particle->getPackingFraction();
            particle->scaleToPackingFractionFull(phi / (scale_factor * scale_factor));
            particle->removeMeanVelocities();
        }
        else {
            particle->countContacts();
            long n_contacts = particle->getContactCount();
            if (avg_ke < avg_ke_thresh && n_contacts > 2 * n_particles && avg_pe > avg_pe_target) {
                // ke < ke thresh and pe > pe target and contacts
                // start bisection if not started yet
                if (D_h < 0) {
                    D_h = D;
                }
                // ke < ke thresh and pe > pe target and contacts
                // expand with bisection if started
                if (D_l > 0) {  // expand with bisection
                    double old_D = D;
                    D_h = D;
                    D = (D_l + D_h) / 2.0;
                    scale_factor = old_D / D;
                    double phi = particle->getPackingFraction();
                    particle->scaleToPackingFractionFull(phi / (scale_factor * scale_factor));
                    particle->removeMeanVelocities();
                }
                // ke < ke thresh and pe > pe target and contacts
                // expand with big steps if bisection not started
                else {  // expand with big steps
                    double old_D = D;
                    D -= sqrt(pe_step / e_c_particle);
                    scale_factor = old_D / D;
                    double phi = particle->getPackingFraction();
                    particle->scaleToPackingFractionFull(phi / (scale_factor * scale_factor));
                    particle->removeMeanVelocities();
                }
            }
            // if ke < ke target and pe < pe target, stop
            else if (avg_ke < avg_ke_target && n_contacts > 2 * n_particles) {
                break;
            }
        }

        compression_step++;
    }
    dynamics_io_manager.log(compression_step, true);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}
#include "../../include/routines/compression.h"

void jam_adam(Particle& particle, Adam& adam, IOManager& io_manager, long num_compression_steps, long num_adam_steps, double avg_pe_target, double avg_pe_diff_target, double packing_fraction_increment, double min_packing_fraction_increment, double mult) {
    particle.initAdamVariables();
    particle.calculateParticleArea();
    double packing_fraction = particle.getPackingFraction();

    long compression_step = 0;
    double avg_pe_past_jamming = 1e-9;  // marks being above jamming (might be too high)
    double avg_pe = 0.0;
    double dof = static_cast<double>(particle.n_dof);
    double last_avg_pe = 0.0;
    double avg_pe_diff = 0.0;
    long adam_step = 0;
    double sign = 1.0;
    long consecutive_compressions = 0;
    while (compression_step < num_compression_steps && avg_pe < avg_pe_past_jamming) {
        adam_step = 0;
        last_avg_pe = 0.0;
        avg_pe_diff = 0.0;
        while (adam_step < num_adam_steps) {
            adam.minimize(adam_step);
            avg_pe = particle.totalPotentialEnergy() / dof / particle.e_c;
            avg_pe_diff = std::abs(avg_pe - last_avg_pe);
            last_avg_pe = avg_pe;
            if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
                break;
            }
            adam_step++;
        }
        if (avg_pe > avg_pe_target) {
            sign = -1.0;
            if (packing_fraction_increment / 2.0 > min_packing_fraction_increment) {
                packing_fraction_increment /= 2.0;
            }
            consecutive_compressions = 0;
        } else if (avg_pe_target * mult > avg_pe && avg_pe > avg_pe_target) {
            std::cout << "jamming complete" << std::endl;
            std::cout << "packing_fraction: " << packing_fraction << std::endl;
            std::cout << "avg_pe: " << avg_pe << std::endl;
            std::cout << "avg_pe_target: " << avg_pe_target << std::endl;
            std::cout << "avg_pe_target * mult: " << avg_pe_target * mult << std::endl;
            break;
        } else {
            sign = 1.0;
            consecutive_compressions++;
            if (consecutive_compressions > 10) {  // make the box bigger
                packing_fraction_increment *= 1.01;
                consecutive_compressions = 0;
            }
        }
        io_manager.log(compression_step);
        particle.scaleToPackingFraction(packing_fraction + packing_fraction_increment * sign);
        packing_fraction = particle.getPackingFraction();
        compression_step++;
    } 
    io_manager.log(compression_step, true);
}
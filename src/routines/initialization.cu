#include "../../include/routines/initialization.h"
#include "../../include/io/io_utils.h"

std::tuple<SwapData2D<double>, SwapData1D<double>, SwapData1D<double>> get_minimal_overlap_disk_positions_and_radii(ConfigDict& config, double overcompression_factor) {
    std::cout << "Running Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;
    double packing_fraction = config["packing_fraction"];
    config["packing_fraction"] = packing_fraction + overcompression_factor;

    Disk disk;
    disk.initializeFromConfig(config);
    disk.initAdamVariables();

    double alpha = 1e-4;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double avg_pe_target = 1e-16;
    double avg_pe_diff_target = 1e-16;
    long num_steps = 1e5;

    AdamConfigDict adam_config;
    adam_config["alpha"] = alpha;
    adam_config["beta1"] = beta1;
    adam_config["beta2"] = beta2;
    adam_config["epsilon"] = epsilon;
    Adam adam(disk, adam_config);

    std::vector<LogGroupConfigDict> log_group_configs = {
        config_from_names_lin_everyN({"step", "PE/N", "phi"}, 1e3, "console"),  // logs to the console
    };
    IOManager io_manager(log_group_configs, disk, &adam, "", 1, true);

    long step = 0;
    double dof = static_cast<double>(disk.n_dof);
    double last_avg_pe = 0.0;
    double avg_pe_diff = 0.0;
    while (step < num_steps) {
        adam.minimize(step);
        io_manager.log(step);
        double avg_pe = disk.totalPotentialEnergy() / dof;
        avg_pe_diff = std::abs(avg_pe - last_avg_pe);
        last_avg_pe = avg_pe;
        if (avg_pe_diff < avg_pe_diff_target || avg_pe < avg_pe_target) {
            break;
        }
        step++;
    }

    // scale the disk positions and radii back down to the target packing fraction
    packing_fraction = config["packing_fraction"];
    disk.scaleToPackingFraction(packing_fraction - overcompression_factor);

    // copy the disk positions and radii and then use that to determine the rigid particle positions
    SwapData2D<double> positions;
    positions.copyFrom(disk.positions);
    SwapData1D<double> radii;
    radii.copyFrom(disk.radii);
    SwapData1D<double> box_size;
    box_size.copyFrom(disk.box_size);
    std::cout << "Finished Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;

    // if the disk was using cell list, reorder the positions and radii by the static_particle_index
    if (disk.using_cell_list) {
        // reorder_array(positions, disk.static_particle_index);
        // reorder_array(radii, disk.static_particle_index);
        ArrayData _static_particle_index = disk.getArrayData("static_particle_index");
        ArrayData _positions = disk.getArrayData("positions");
        ArrayData _radii = disk.getArrayData("radii");
        reorder_array(_positions, _static_particle_index);
        reorder_array(_radii, _static_particle_index);
        // get the data from the reordered arrays
        auto& _positions_data = get_2d_data<double>(_positions);
        auto& _radii_data = get_1d_data<double>(_radii);
        // copy the data back to the original arrays
        const thrust::host_vector<double>& positions_data_x = _positions_data.first;
        const thrust::host_vector<double>& positions_data_y = _positions_data.second;
        const thrust::host_vector<double>& radii_data = _radii_data;
        positions.setData(positions_data_x, positions_data_y);
        radii.setData(radii_data);
    }

    return std::make_tuple(positions, radii, box_size);
}

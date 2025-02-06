#include "../../include/routines/initialization.h"
#include "../../include/routines/minimization.h"
#include "../../include/io/io_utils.h"

std::tuple<thrust::host_vector<double>, thrust::host_vector<double>, thrust::host_vector<double>, thrust::host_vector<double>> get_minimal_overlap_disk_positions_and_radii(ConfigDict& config, thrust::host_vector<double> positions_x, thrust::host_vector<double> positions_y, thrust::host_vector<double> radii, thrust::host_vector<double> box_size, double overcompression) {
    std::cout << "Running Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;
    double packing_fraction = config.at("packing_fraction").get<double>();
    packing_fraction += overcompression;
    config["packing_fraction"] = packing_fraction;

    Disk disk;
    disk.initializeFromConfig(config, false);
    disk.positions.setData(positions_x, positions_y);
    disk.radii.setData(radii);
    disk.box_size.setData(box_size);
    disk.scaleToPackingFraction(packing_fraction);
    disk.setupNeighbors(config);
    minimizeAdam(disk);
    packing_fraction -= overcompression;
    config["packing_fraction"] = packing_fraction;
    disk.scaleToPackingFraction(config["packing_fraction"]);

    // copy the disk positions and radii and then use that to determine the rigid particle positions
    positions_x = disk.positions.getDataX();
    positions_y = disk.positions.getDataY();
    radii = disk.radii.getData();
    box_size = disk.box_size.getData();
    std::cout << "Finished Routine: get_minimal_overlap_disk_positions_and_radii" << std::endl;

    // if the disk was using cell list, reorder the positions and radii by the static_particle_index
    if (disk.using_cell_list) {
        // reorder the data arrays according to the static_particle_index
        thrust::host_vector<long> static_particle_index = disk.static_particle_index.getData();

        auto zipped_values = thrust::make_zip_iterator(
            thrust::make_tuple(positions_x.begin(), positions_y.begin(), radii.begin())
        );

        // Sort them in one pass by static_particle_index
        thrust::sort_by_key(
            static_particle_index.begin(),
            static_particle_index.end(),
            zipped_values
        );
    }
    return std::make_tuple(positions_x, positions_y, radii, box_size);
}

#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include "../../include/data/data_array.h"

int main() {
    // Creating a device 1D data object of double
    Data1D<double, ExecutionSpace::Device> device_object;
    device_object.resize(3);
    device_object.fill(3.14);

    // Convert to host
    auto host_object = device_object.to_host();

    // Access the pointer on the host side
    double* host_ptr = host_object.vector_pointer();

    Data1D<double, ExecutionSpace::Host> new_host_object;
    new_host_object.resize(3);
    new_host_object.fillRandomNormal(0.0, 1.0, 123);

    // Convert to device
    auto new_device_object = new_host_object.to_device();

    // Get the data
    auto data = new_device_object.getData();

    // Print the data
    std::cout << "Original data:" << std::endl;
    for (int i = 0; i < data.size(); i++) {
        std::cout << data[i] << std::endl;
    }

    // Create a 2D data object
    Data2D<double, ExecutionSpace::Device> device_2d_object;
    device_2d_object.resize(3);
    device_2d_object.fillRandomNormal(0.0, 1.0, 123);

    // Get the data
    auto [host_x, host_y] = device_2d_object.getData();

    // Print the data
    std::cout << "Original data:" << std::endl;
    for (int i = 0; i < host_x.size(); i++) {
        std::cout << host_x[i] << " " << host_y[i] << std::endl;
    }

    // Save the data to a file
    std::string filename = "/home/mmccraw/dev/dpCUDA/scripts/tests/test_data/array_data/data_2d.dat";
    device_2d_object.save(filename);

    // Load the data from a file
    Data2D<double, ExecutionSpace::Device> loaded_2d_object;
    loaded_2d_object.load(filename);

    // Get the data
    auto [loaded_x, loaded_y] = loaded_2d_object.getData();

    // Print the data
    std::cout << "Loaded data:" << std::endl;
    for (int i = 0; i < loaded_x.size(); i++) {
        std::cout << loaded_x[i] << " " << loaded_y[i] << std::endl;
    }

    // Reorder the data
    thrust::host_vector<long> index(loaded_x.size());
    for (int i = 0; i < index.size(); i++) {
        index[i] = i;
    }
    index[0] = 2;
    index[1] = 0;
    index[2] = 1;
    loaded_2d_object.reorder(index);

    // Get the data
    auto [reordered_x, reordered_y] = loaded_2d_object.getData();

    // Print the data
    std::cout << "Reordered data:" << std::endl;
    for (int i = 0; i < reordered_x.size(); i++) {
        std::cout << reordered_x[i] << " " << reordered_y[i] << std::endl;
    }

    // Scale the data
    loaded_2d_object.scale(0.5);

    // Get the data
    auto [half_scaled_x, half_scaled_y] = loaded_2d_object.getData();

    // Print the data
    std::cout << "Halved data:" << std::endl;
    for (int i = 0; i < half_scaled_x.size(); i++) {
        std::cout << half_scaled_x[i] << " " << half_scaled_y[i] << std::endl;
    }

    // Create a swappable data object
    SwappableData1D<double, ExecutionSpace::Device> device_swappable_object;
    device_swappable_object.resize(3);
    device_swappable_object.fillRandomNormal(0.0, 1.0, 123);

    // Swap the data
    device_swappable_object.swapData();
    device_swappable_object.swapData();

    // Get the data
    auto swapped_data = device_swappable_object.getData();

    // Print the data
    std::cout << "Swapped data:" << std::endl;
    for (int i = 0; i < swapped_data.size(); i++) {
        std::cout << swapped_data[i] << std::endl;
    }

    // Save the data to a file
    std::string swapped_filename = "/home/mmccraw/dev/dpCUDA/scripts/tests/test_data/array_data/swapped_data.dat";
    device_swappable_object.save(swapped_filename);

    // Load the data from a file
    SwappableData1D<double, ExecutionSpace::Device> loaded_swappable_object;
    loaded_swappable_object.load(swapped_filename);

    // Swap it
    loaded_swappable_object.swapData();
    loaded_swappable_object.swapData();

    // Get the data
    auto loaded_swapped_data = loaded_swappable_object.getData();

    // Print the data
    std::cout << "Loaded swapped data:" << std::endl;
    for (int i = 0; i < loaded_swapped_data.size(); i++) {
        std::cout << loaded_swapped_data[i] << std::endl;
    }

    // Create a 2D swappable data object
    SwappableData2D<double, ExecutionSpace::Device> device_swappable_2d_object;
    device_swappable_2d_object.resize(3);
    device_swappable_2d_object.fillRandomNormal(0.0, 1.0, 123);

    // Swap the data
    device_swappable_2d_object.swapData();
    device_swappable_2d_object.swapData();

    // Get the data
    auto [swapped_x, swapped_y] = device_swappable_2d_object.getData();

    // Print the data
    std::cout << "Swapped data:" << std::endl;
    for (int i = 0; i < swapped_x.size(); i++) {
        std::cout << swapped_x[i] << " " << swapped_y[i] << std::endl;
    }

    // Save the data to a file
    std::string swapped_2d_filename = "/home/mmccraw/dev/dpCUDA/scripts/tests/test_data/array_data/swapped_2d_data.dat";
    device_swappable_2d_object.save(swapped_2d_filename);

    // Load the data from a file
    SwappableData2D<double, ExecutionSpace::Device> loaded_swappable_2d_object;
    loaded_swappable_2d_object.load(swapped_2d_filename);

    // Also load a 2d non-swappable host data object using the same file
    Data2D<double, ExecutionSpace::Host> non_swapped_loaded_2d_object;
    non_swapped_loaded_2d_object.load(swapped_2d_filename);

    // Get the data
    auto [loaded_swapped_x, loaded_swapped_y] = loaded_swappable_2d_object.getData();
    auto [non_swapped_x, non_swapped_y] = non_swapped_loaded_2d_object.getData();

    // Print the data
    std::cout << "Loaded swapped data:" << std::endl;
    for (int i = 0; i < loaded_swapped_x.size(); i++) {
        std::cout << loaded_swapped_x[i] << " " << loaded_swapped_y[i] << std::endl;
    }
    std::cout << "Non-swapped loaded data:" << std::endl;
    for (int i = 0; i < non_swapped_x.size(); i++) {
        std::cout << non_swapped_x[i] << " " << non_swapped_y[i] << std::endl;
    }

    // Using the non-swapped host data, create a swappable host data object
    SwappableData2D<double, ExecutionSpace::Host> host_swappable_2d_object;
    host_swappable_2d_object.setData(non_swapped_x, non_swapped_y);

    // Swap the data
    host_swappable_2d_object.swapData();
    host_swappable_2d_object.swapData();

    // Get the data
    auto [new_swapped_x, new_swapped_y] = host_swappable_2d_object.getData();

    // Print the data
    std::cout << "Swapped data:" << std::endl;
    for (int i = 0; i < new_swapped_x.size(); i++) {
        std::cout << new_swapped_x[i] << " " << new_swapped_y[i] << std::endl;
    }

    // Scale the data
    host_swappable_2d_object.scale(2.0, 3.0);

    // Get the data
    auto [scaled_x, scaled_y] = host_swappable_2d_object.getData();

    // Print the data
    std::cout << "Scaled data:" << std::endl;
    for (int i = 0; i < scaled_x.size(); i++) {
        std::cout << scaled_x[i] << " " << scaled_y[i] << std::endl;
    }
}
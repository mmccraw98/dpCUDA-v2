#ifndef ARRAY_DATA_H
#define ARRAY_DATA_H

#include <thrust/host_vector.h>
#include <string>
#include <variant>
#include <utility>
#include <array>

enum class DataType { Double, Long };

// for quickly getting the data out of the data1d/2d classes (out of the particle class) and into the io module
// it might be better if the data1d/2d classes were templated based on something like this so we dont have to introduce a new type
// for the sake of io
// but then we would need to do a lot of work changing the data1d/2d classes which is painful
// one could also argue that the data1d/2d classes should be performant and not worry about io
// whereas the arraydata could be specifically for io where performance is not as important
struct ArrayData {
    std::string name;
    DataType type;
    std::array<long, 2> size;  // Now holds the size array from Data1D/2D
    std::string index_array_name = "";  // the name of the index array used for reordering this data (if needed)

    // Data can be one of the following types
    std::variant<
        thrust::host_vector<double>,                                        // For 1D double arrays
        thrust::host_vector<long>,                                          // For 1D long arrays
        std::pair<thrust::host_vector<double>, thrust::host_vector<double>>, // For 2D double arrays
        std::pair<thrust::host_vector<long>, thrust::host_vector<long>>      // For 2D long arrays (if any)
    > data;
};


#endif // ARRAY_DATA_H
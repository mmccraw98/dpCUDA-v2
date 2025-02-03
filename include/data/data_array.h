#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <type_traits>
#include <fstream>
#include <iomanip>

// Simple enum to distinguish between host and device memory.
enum class ExecutionSpace {
    Host,
    Device
};

//------------------------------------------------------------
// 1D Data Object
//------------------------------------------------------------
template <typename T, ExecutionSpace ES>
class Data1D 
{
public:
    // Conditionally pick a thrust host_vector or device_vector.
    using VectorType = std::conditional_t<
        ES == ExecutionSpace::Host,
        thrust::host_vector<T>,
        thrust::device_vector<T>
    >;

    // Publicly accessible vector (either host_vector or device_vector).
    VectorType vector;

    Data1D() = default;

    // Return a pointer to the underlying storage.
    // For a host_vector, this is just vector.data().
    // For a device_vector, we use thrust::raw_pointer_cast(vector.data()).
    T* vector_pointer()
    {
        if constexpr (ES == ExecutionSpace::Host) {
            return vector.data();
        } else {
            return thrust::raw_pointer_cast(vector.data());
        }
    }

    // Resize the vector to new_size.
    void resize(long new_size)
    {
        vector.resize(static_cast<size_t>(new_size));
    }

    // Fill the vector with a given value.
    void fill(const T& value)
    {
        thrust::fill(vector.begin(), vector.end(), value);
    }

    // Resize and then fill.
    void resizeAndFill(long new_size, const T& value)
    {
        resize(new_size);
        fill(value);
    }

    // Clear the contents of the vector.
    void clear()
    {
        vector.clear();
    }

    // Set the data from a host_vector<T>. 
    // If this object is device-based, it will do device_vector = host_vector automatically.
    void setData(const thrust::host_vector<T>& h_data)
    {
        if constexpr (ES == ExecutionSpace::Host) {
            vector = h_data; 
        } else {
            vector = h_data; // device_vector can be assigned from a host_vector
        }
    }

    // Get the data as a host_vector<T>. 
    // If we are host-based, returns a copy of our host_vector. 
    // If we are device-based, returns a copy transferred back to the host.
    thrust::host_vector<T> getData() const
    {
        if constexpr (ES == ExecutionSpace::Host) {
            // Return a copy for consistency
            return vector; 
        } else {
            // Assigning a device_vector to a host_vector copies device->host
            thrust::host_vector<T> h_data = vector; 
            return h_data;
        }
    }

    // Return a new Data1D on the host, copying data over if needed.
    Data1D<T, ExecutionSpace::Host> to_host() const
    {
        Data1D<T, ExecutionSpace::Host> host_copy;
        host_copy.setData(getData());
        return host_copy;
    }

    // Return a new Data1D on the device, copying data over if needed.
    Data1D<T, ExecutionSpace::Device> to_device() const
    {
        Data1D<T, ExecutionSpace::Device> device_copy;
        device_copy.setData(getData());
        return device_copy;
    }

    // Fill the vector with random normal values.
    void fillRandomNormal(double mean, double std_dev, long seed)
    {
        // Example usage of your prior snippet:
        // thrust::counting_iterator<long> index_sequence_begin(seed + seed_offset);
        // If you don’t need a seed_offset, just do:
        thrust::counting_iterator<long> index_sequence_begin(seed);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + vector.size(),
                          vector.begin(),
                          RandomNormal(mean, std_dev, seed));
    }

    // Fill the vector with random uniform values.
    void fillRandomUniform(double min_val, double max_val, long seed)
    {
        thrust::counting_iterator<long> index_sequence_begin(seed);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + vector.size(),
                          vector.begin(),
                          RandomUniform(min_val, max_val, seed));
    }

    // Save the data to a file
    void save(const std::string& filename, int precision = DECIMAL_PRECISION) const
    {
        // Pull data to host if needed
        auto h_data = getData();

        std::ofstream ofs(filename);
        if(!ofs.is_open()){
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        ofs << std::fixed << std::setprecision(precision);

        for(const auto& val : h_data) {
            ofs << val << "\n";
        }

        ofs.close();
    }

    // Load the data from a file
    void load(const std::string& filename)
    {
        std::ifstream ifs(filename);
        if(!ifs.is_open()){
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        thrust::host_vector<T> h_data;
        T value;
        while(ifs >> value) {
            h_data.push_back(value);
        }
        ifs.close();

        // Now store into our internal vector
        setData(h_data);
    }

    // Reorder the data in the vector.
    template <typename IndexVector>
    void reorder(const IndexVector& index)
    {
        // 1) Make sure we have a vector of the same execution space as `vector`
        //    We'll call this "index_copy" to do the actual scatter.
        using IndexVectorType = std::conditional_t<
            ES == ExecutionSpace::Host,
            thrust::host_vector<long>,
            thrust::device_vector<long>
        >;

        IndexVectorType index_copy;
        // Copy from input to index_copy (device->device or host->device if needed)
        index_copy = index;

        // 2) Create a new_data vector
        VectorType new_data(vector.size());

        // 3) Perform scatter: old data (vector) -> new_data, using index_copy as map
        //    "data[i] goes to new_data[index_copy[i]]"
        thrust::scatter(vector.begin(), vector.end(), index_copy.begin(), new_data.begin());

        // 4) Swap old and new
        vector.swap(new_data);
    }
};

//------------------------------------------------------------
// 2D Data Object
//  - Holds two vectors x, y (either on Host or Device).
//------------------------------------------------------------
template <typename T, ExecutionSpace ES>
class Data2D
{
public:
    // Conditionally pick a thrust host_vector or device_vector.
    using VectorType = std::conditional_t<
        ES == ExecutionSpace::Host,
        thrust::host_vector<T>,
        thrust::device_vector<T>
    >;

    // Two vectors: x and y
    VectorType x;
    VectorType y;

    Data2D() = default;

    // Return a pointer to x.
    T* x_pointer()
    {
        if constexpr (ES == ExecutionSpace::Host) {
            return x.data();
        } else {
            return thrust::raw_pointer_cast(x.data());
        }
    }

    // Return a pointer to y.
    T* y_pointer()
    {
        if constexpr (ES == ExecutionSpace::Host) {
            return y.data();
        } else {
            return thrust::raw_pointer_cast(y.data());
        }
    }

    // Resize both x and y to new_size.
    void resize(long new_size)
    {
        x.resize(static_cast<size_t>(new_size));
        y.resize(static_cast<size_t>(new_size));
    }

    // Fill both x and y with the specified value.
    void fill(const T& value)
    {
        thrust::fill(x.begin(), x.end(), value);
        thrust::fill(y.begin(), y.end(), value);
    }

    // Resize and then fill both arrays.
    void resizeAndFill(long new_size, const T& value)
    {
        resize(new_size);
        fill(value);
    }

    // Clear both vectors.
    void clear()
    {
        x.clear();
        y.clear();
    }

    // Set data from two host vectors.
    void setData(const thrust::host_vector<T>& hx, const thrust::host_vector<T>& hy)
    {
        if constexpr (ES == ExecutionSpace::Host) {
            x = hx;
            y = hy;
        } else {
            x = hx; // device_vector = host_vector
            y = hy; 
        }
    }

    // Return two host vectors of x, y data.
    // If device-based, copies back from device to host.
    std::pair<thrust::host_vector<T>, thrust::host_vector<T>> getData() const
    {
        if constexpr (ES == ExecutionSpace::Host) {
            return std::make_pair(x, y); 
        } else {
            return std::make_pair(thrust::host_vector<T>(x), thrust::host_vector<T>(y));
        }
    }

    // Create a host-based copy of this Data2D.
    Data2D<T, ExecutionSpace::Host> to_host() const
    {
        Data2D<T, ExecutionSpace::Host> host_copy;
        auto [hx, hy] = getData();
        host_copy.setData(hx, hy);
        return host_copy;
    }

    // Create a device-based copy of this Data2D.
    Data2D<T, ExecutionSpace::Device> to_device() const
    {
        Data2D<T, ExecutionSpace::Device> device_copy;
        auto [hx, hy] = getData();
        device_copy.setData(hx, hy);
        return device_copy;
    }

    // Fill both x and y with random normal values.
    void fillRandomNormal(double mean, double std_dev, long seed)
    {
        // Fill x using seed
        thrust::counting_iterator<long> index_sequence_begin(seed);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + x.size(),
                          x.begin(),
                          RandomNormal(mean, std_dev, seed));

        // Fill y using seed + x.size() so that it does not overlap
        thrust::counting_iterator<long> index_sequence_begin_y(seed + static_cast<long>(x.size()));
        thrust::transform(index_sequence_begin_y,
                          index_sequence_begin_y + y.size(),
                          y.begin(),
                          RandomNormal(mean, std_dev, seed));
    }

    // Fill both x and y with random uniform values.
    void fillRandomUniform(double min_val, double max_val, long seed)
    {
        // Fill x using seed
        thrust::counting_iterator<long> index_sequence_begin(seed);
        thrust::transform(index_sequence_begin,
                          index_sequence_begin + x.size(),
                          x.begin(),
                          RandomUniform(min_val, max_val, seed));

        // Fill y using seed + x.size()
        thrust::counting_iterator<long> index_sequence_begin_y(seed + static_cast<long>(x.size()));
        thrust::transform(index_sequence_begin_y,
                          index_sequence_begin_y + y.size(),
                          y.begin(),
                          RandomUniform(min_val, max_val, seed));
    }

    // Save the data to a file
    void save(const std::string& filename, int precision = DECIMAL_PRECISION) const
    {
        // Pull data into host vectors
        auto [hx, hy] = getData();

        std::ofstream ofs(filename);
        if(!ofs.is_open()){
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        ofs << std::fixed << std::setprecision(precision);

        // For each element, write "x[i]\t y[i]"
        for(size_t i = 0; i < hx.size(); i++){
            ofs << hx[i] << "\t" << hy[i] << "\n";
        }

        ofs.close();
    }

    // Load the data from a file
    void load(const std::string& filename)
    {
        std::ifstream ifs(filename);
        if(!ifs.is_open()){
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        thrust::host_vector<T> hx_data;
        thrust::host_vector<T> hy_data;

        T xval, yval;
        while(ifs >> xval >> yval){
            hx_data.push_back(xval);
            hy_data.push_back(yval);
        }
        ifs.close();

        setData(hx_data, hy_data);
    }

    // Reorder the data in the vector.
    template <typename IndexVector>
    void reorder(const IndexVector& index)
    {
        // We'll reorder both x and y using the same index map.
        using IndexVectorType = std::conditional_t<
            ES == ExecutionSpace::Host,
            thrust::host_vector<long>,
            thrust::device_vector<long>
        >;

        // Copy from input to index_copy as needed
        IndexVectorType index_copy = index;

        // Create new_x, new_y
        VectorType new_x(x.size());
        VectorType new_y(y.size());

        // scatter from x -> new_x
        thrust::scatter(x.begin(), x.end(), index_copy.begin(), new_x.begin());
        // scatter from y -> new_y
        thrust::scatter(y.begin(), y.end(), index_copy.begin(), new_y.begin());

        // swap them
        x.swap(new_x);
        y.swap(new_y);
    }
};

//------------------------------------------------------------
// Swappable Data1D Object
//------------------------------------------------------------
template <typename T, ExecutionSpace ES>
class SwappableData1D : public Data1D<T, ES>
{
public:
    using Base       = Data1D<T, ES>;
    using VectorType = typename Base::VectorType;

    VectorType temp_vector;  // second "temp" array

    SwappableData1D() : Base() {}

    // Ensure both main vector and temp vector get resized
    void resize(long new_size)
    {
        Base::resize(new_size);   // handles Base::vector
        temp_vector.resize(static_cast<size_t>(new_size));
    }

    // Swap the underlying data pointers (instant O(1) swap)
    void swapData()
    {
        Base::vector.swap(temp_vector);
    }

    // Overriding clear() so we also clear temp_vector
    void clear()
    {
        Base::clear();       // clears Base::vector
        temp_vector.clear(); // also clear temp
    }

    // Overriding load() so that temp_vector has the same size as the main vector
    void load(const std::string& filename)
    {
        Base::load(filename);              // loads into Base::vector
        temp_vector.resize(Base::vector.size()); // ensure temp_vector matches
    }
};

//------------------------------------------------------------
// Swappable Data2D Object
//------------------------------------------------------------
template <typename T, ExecutionSpace ES>
class SwappableData2D : public Data2D<T, ES>
{
public:
    using Base       = Data2D<T, ES>;
    using VectorType = typename Base::VectorType;

    VectorType temp_x;
    VectorType temp_y;

    SwappableData2D() : Base() {}

    // Resize both main and temp arrays
    void resize(long new_size)
    {
        Base::resize(new_size);    // handles x, y
        temp_x.resize(new_size);
        temp_y.resize(new_size);
    }

    // Swap the underlying data pointers
    void swapData()
    {
        Base::x.swap(temp_x);
        Base::y.swap(temp_y);
    }

    // Clear both main and temp
    void clear()
    {
        Base::clear();  // clears x, y
        temp_x.clear();
        temp_y.clear();
    }

    // Load main arrays, then ensure temp_x/temp_y match
    void load(const std::string& filename)
    {
        Base::load(filename);  // loads x,y from file
        temp_x.resize(Base::x.size());
        temp_y.resize(Base::y.size());
    }
};
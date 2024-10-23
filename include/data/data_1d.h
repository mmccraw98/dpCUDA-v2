#pragma once

#include "../../include/constants.h"
#include "../../include/functors.h"

#include <array>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>


// base

template <typename T>
class BaseData1D {
public:
    // Members
    std::array<long, 2> size = {0, 1};
    thrust::device_vector<T> d_vec;
    T* d_ptr = nullptr;

    // Constructors
    BaseData1D();
    BaseData1D(long N, T value = T());

    // Common functions
    void resize(long new_size);
    void fill(T value);
    void resizeAndFill(long new_size, T value);
    void clear();
    void setData(const thrust::host_vector<T>& host_data);
    thrust::host_vector<T> getData() const;
};

// derived for types

template <typename T>
class Data1D : public BaseData1D<T> {
public:
    Data1D();
    Data1D(long N, T value = T());
};


// specialization for double

template <>
class Data1D<double> : public BaseData1D<double> {
public:
    Data1D();
    Data1D(long N, double value = 0.0);
    void scale(double scale_factor);
    void fillRandomNormal(double mean, double std_dev, long seed_offset, long seed);
    void fillRandomUniform(double min, double max, long seed_offset, long seed);
};

// swappable

template <typename T>
class SwapData1D : public Data1D<T> {
    public:
        thrust::device_vector<T> d_temp_vec;
        T* d_temp_ptr = nullptr;

        SwapData1D();
        SwapData1D(long N, T value = T());

        void resize(long new_size);
        void resizeAndFill(long new_size, T value);
        void fill(T value);
        void clear();
        void swap();
};
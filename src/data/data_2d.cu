
#include "../../include/data/data_2d.h"
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>

// Constructors for BaseData2D
template <typename T, typename Data1DType>
BaseData2D<T, Data1DType>::BaseData2D() = default;

template <typename T, typename Data1DType>
BaseData2D<T, Data1DType>::BaseData2D(long N, T value_x, T value_y)
    : size{N, 2}, x(N, value_x), y(N, value_y) {}

// Resize
template <typename T, typename Data1DType>
void BaseData2D<T, Data1DType>::resize(long new_size) {
    size[0] = new_size;
    x.resize(new_size);
    y.resize(new_size);
}

// Resize and fill
template <typename T, typename Data1DType>
void BaseData2D<T, Data1DType>::resizeAndFill(long new_size, T value_x, T value_y) {
    size[0] = new_size;
    x.resizeAndFill(new_size, value_x);
    y.resizeAndFill(new_size, value_y);
}

// Fill
template <typename T, typename Data1DType>
void BaseData2D<T, Data1DType>::fill(T value_x, T value_y) {
    x.fill(value_x);
    y.fill(value_y);
}

// Clear
template <typename T, typename Data1DType>
void BaseData2D<T, Data1DType>::clear() {
    x.clear();
    y.clear();
}

// Set data
template <typename T, typename Data1DType>
void BaseData2D<T, Data1DType>::setData(const thrust::host_vector<T>& host_data_x, const thrust::host_vector<T>& host_data_y) {
    x.setData(host_data_x);
    y.setData(host_data_y);
}

// Get data
template <typename T, typename Data1DType>
thrust::host_vector<T> BaseData2D<T, Data1DType>::getDataX() const {
    return x.getData();
}

template <typename T, typename Data1DType>
thrust::host_vector<T> BaseData2D<T, Data1DType>::getDataY() const {
    return y.getData();
}

// Data2D constructors
template <typename T, typename Data1DType>
Data2D<T, Data1DType>::Data2D() = default;

template <typename T, typename Data1DType>
Data2D<T, Data1DType>::Data2D(long N, T value_x, T value_y)
    : BaseData2D<T, Data1DType>(N, value_x, value_y) {}

// Specialized Data2D<double, Data1DType> methods
template <typename Data1DType>
Data2D<double, Data1DType>::Data2D() = default;

template <typename Data1DType>
Data2D<double, Data1DType>::Data2D(long N, double value_x, double value_y)
    : BaseData2D<double, Data1DType>(N, value_x, value_y) {}

template <typename Data1DType>
void Data2D<double, Data1DType>::scale(double scale_factor_x, double scale_factor_y) {
    this->x.scale(scale_factor_x);
    this->y.scale(scale_factor_y);
}

template <typename Data1DType>
void Data2D<double, Data1DType>::fillRandomNormal(double mean_x, double std_dev_x, double mean_y, double std_dev_y, long seed_offset, long seed) {
    this->x.fillRandomNormal(mean_x, std_dev_x, seed_offset, seed);
    this->y.fillRandomNormal(mean_y, std_dev_y, seed_offset + this->x.size[0], seed);
}

template <typename Data1DType>
void Data2D<double, Data1DType>::fillRandomUniform(double min_x, double max_x, double min_y, double max_y, long seed_offset, long seed) {
    this->x.fillRandomUniform(min_x, max_x, seed_offset, seed);
    this->y.fillRandomUniform(min_y, max_y, seed_offset + this->x.size[0], seed);
}

// SwapData2D implementation
template <typename T>
SwapData2D<T>::SwapData2D() = default;

template <typename T>
SwapData2D<T>::SwapData2D(long N, T value_x, T value_y)
    : Data2D<T, SwapData1D<T>>(N, value_x, value_y) {}

template <typename T>
void SwapData2D<T>::swap() {
    this->x.swap();
    this->y.swap();
}

// Explicit template instantiation for BaseData2D with Data1D<T>
template class BaseData2D<long, Data1D<long>>;
template class BaseData2D<double, Data1D<double>>;

// Explicit template instantiation for BaseData2D with SwapData1D<T>
template class BaseData2D<long, SwapData1D<long>>;
template class BaseData2D<double, SwapData1D<double>>;

// Explicit template instantiation for Data2D with Data1D<T>
template class Data2D<long, Data1D<long>>;
template class Data2D<double, Data1D<double>>;

// Explicit template instantiation for Data2D with SwapData1D<T>
template class Data2D<long, SwapData1D<long>>;
template class Data2D<double, SwapData1D<double>>;

// Explicit template instantiation for SwapData2D
template class SwapData2D<long>;
template class SwapData2D<double>;
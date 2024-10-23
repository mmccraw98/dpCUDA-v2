// BaseData1D.cu

#include "../../include/data/data_1d.h"
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/random.h>

// Constructors
template <typename T>
BaseData1D<T>::BaseData1D() = default;

template <typename T>
BaseData1D<T>::BaseData1D(long N, T value) : size{N, 1}, d_vec(N, value) {
    d_ptr = d_vec.data().get();
}

// Resize
template <typename T>
void BaseData1D<T>::resize(long new_size) {
    size[0] = new_size;
    d_vec.resize(new_size);
    d_ptr = d_vec.data().get();
}

// Fill
template <typename T>
void BaseData1D<T>::fill(T value) {
    thrust::fill(d_vec.begin(), d_vec.end(), value);
}

// Resize and fill
template <typename T>
void BaseData1D<T>::resizeAndFill(long new_size, T value) {
    resize(new_size);
    fill(value);
}

// Clear
template <typename T>
void BaseData1D<T>::clear() {
    d_vec.clear();
    d_ptr = nullptr;
}

// Set data from host
template <typename T>
void BaseData1D<T>::setData(const thrust::host_vector<T>& host_data) {
    thrust::copy(host_data.begin(), host_data.end(), d_vec.begin());
}

// Get data to host
template <typename T>
thrust::host_vector<T> BaseData1D<T>::getData() const {
    thrust::host_vector<T> host_data(size[0]);
    thrust::copy(d_vec.begin(), d_vec.end(), host_data.begin());
    return host_data;
}

// Copy from another BaseData1D
template <typename T>
void BaseData1D<T>::copyFrom(const BaseData1D<T>& other) {
    resize(other.size[0]);
    thrust::copy(other.d_vec.begin(), other.d_vec.end(), d_vec.begin());
}

// Specializations

// template instantiation
template <typename T>
Data1D<T>::Data1D() = default;

template <typename T>
Data1D<T>::Data1D(long N, T value) : BaseData1D<T>(N, value) {}

// double
Data1D<double>::Data1D() = default;

Data1D<double>::Data1D(long N, double value) : BaseData1D<double>(N, value) {}

void Data1D<double>::scale(double scale_factor) {
    thrust::transform(d_vec.begin(), d_vec.end(), thrust::make_constant_iterator(scale_factor), d_vec.begin(), thrust::multiplies<double>());
}

void Data1D<double>::fillRandomNormal(double mean, double std_dev, long seed_offset, long seed) {
    thrust::counting_iterator<long> index_sequence_begin(seed + seed_offset);
    thrust::transform(index_sequence_begin, index_sequence_begin + d_vec.size(), d_vec.begin(), RandomNormal(mean, std_dev, seed));
}

void Data1D<double>::fillRandomUniform(double min, double max, long seed_offset, long seed) {
    thrust::counting_iterator<long> index_sequence_begin(seed + seed_offset);
    thrust::transform(index_sequence_begin, index_sequence_begin + d_vec.size(), d_vec.begin(), RandomUniform(min, max, seed));
}

// swappable

template <typename T>
SwapData1D<T>::SwapData1D() : Data1D<T>(), d_temp_vec(), d_temp_ptr(nullptr) {
    // Initialize d_temp_vec to match d_vec size
    d_temp_vec.resize(this->d_vec.size());
    d_temp_ptr = d_temp_vec.data().get();
}

template <typename T>
SwapData1D<T>::SwapData1D(long N, T value) : Data1D<T>(N, value), d_temp_vec(N, value) {
    d_temp_ptr = d_temp_vec.data().get();
}

template <typename T>
void SwapData1D<T>::resize(long new_size) {
    Data1D<T>::resize(new_size);
    d_temp_vec.resize(new_size);
    d_temp_ptr = d_temp_vec.data().get();
}

template <typename T>
void SwapData1D<T>::resizeAndFill(long new_size, T value) {
    Data1D<T>::resizeAndFill(new_size, value);
    d_temp_vec.resize(new_size);
    thrust::fill(d_temp_vec.begin(), d_temp_vec.end(), value);
    d_temp_ptr = d_temp_vec.data().get();
}

template <typename T>
void SwapData1D<T>::fill(T value) {
    Data1D<T>::fill(value);
    thrust::fill(d_temp_vec.begin(), d_temp_vec.end(), value);
}

template <typename T>
void SwapData1D<T>::clear() {
    Data1D<T>::clear();
    d_temp_vec.clear();
    d_temp_ptr = nullptr;
}

template <typename T>
void SwapData1D<T>::swap() {
    thrust::swap(this->d_vec, d_temp_vec);
    std::swap(this->d_ptr, d_temp_ptr);
}

// Explicit template instantiation
template class BaseData1D<long>;
template class BaseData1D<double>;
template class Data1D<long>;
template class Data1D<double>;
template class SwapData1D<double>;
template class SwapData1D<long>;

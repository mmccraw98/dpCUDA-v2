#ifndef DATA_2D_H
#define DATA_2D_H

#include "data_1d.h"

// BaseData2D with Data1DType as a template parameter
template <typename T, typename Data1DType = Data1D<T>>
class BaseData2D {
    public:
        Data1DType x;
        Data1DType y;
        std::array<long, 2> size = {0, 2};

        BaseData2D();
        BaseData2D(long N, T value_x = T(), T value_y = T());

        void resize(long new_size);
        void resizeAndFill(long new_size, T value_x, T value_y);
        void fill(T value_x, T value_y);
        void clear();
        void setData(const thrust::host_vector<T>& host_data_x, const thrust::host_vector<T>& host_data_y);
        thrust::host_vector<T> getDataX() const;
        thrust::host_vector<T> getDataY() const;
};

// Data2D accepts Data1DType as a template parameter
template <typename T, typename Data1DType = Data1D<T>>
class Data2D : public BaseData2D<T, Data1DType> {
    public:
        Data2D();
        Data2D(long N, T value_x = T(), T value_y = T());
};

// Specialized Data2D for double with additional methods
template <typename Data1DType>
class Data2D<double, Data1DType> : public BaseData2D<double, Data1DType> {
    public:
        Data2D();
        Data2D(long N, double value_x = 0.0, double value_y = 0.0);
        void scale(double scale_factor_x, double scale_factor_y);
        void fillRandomNormal(double mean_x, double std_dev_x, double mean_y, double std_dev_y, long seed_offset, long seed);
        void fillRandomUniform(double min_x, double max_x, double min_y, double max_y, long seed_offset, long seed);
};

// SwapData2D uses SwapData1D<T> for x and y
template <typename T>
class SwapData2D : public Data2D<T, SwapData1D<T>> {
    public:
        SwapData2D();
        SwapData2D(long N, T value_x = T(), T value_y = T());
        void swap();
};

#endif // DATA_2D_H
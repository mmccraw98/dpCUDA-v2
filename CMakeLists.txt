# Minimum required version of CMake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name and supported languages
project(dpCUDAv2 LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set CUDA directory (if needed)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

# Add include directories (including CUDA's include directory)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/data
    ${CMAKE_SOURCE_DIR}/include/integrator
    ${CMAKE_SOURCE_DIR}/include/particle
    ${CMAKE_SOURCE_DIR}/include/kernels
    ${CMAKE_SOURCE_DIR}/include/io
    ${CMAKE_SOURCE_DIR}/include/utils
    ${CMAKE_SOURCE_DIR}/include/routines
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Find the nlohmann_json package
find_package(nlohmann_json 3.2.0 REQUIRED)

# CC compiler flags (G++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fdevirtualize -MMD -MP -flto")

# NVCC compiler flags (CUDA compiler)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress=550 -Wno-deprecated-gpu-targets -Xcompiler -fdevirtualize -MMD -MP -arch=sm_89")

# Libraries for C++ (e.g., linking stdc++fs for filesystem support)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Define a STATIC library for shared sources to ensure proper CUDA linkage
add_library(common_sources STATIC
    src/data/data_1d.cu
    src/data/data_2d.cu
    src/kernels/kernels.cu
    src/particle/config.cu
    src/particle/particle.cu
    src/particle/particle_factory.cu
    src/io/orchestrator.cu
    src/io/io_manager.cu
    src/io/base_log_groups.cu
    src/io/console_log.cu
    src/io/energy_log.cu
    src/io/state_log.cu
    src/integrator/integrator.cu
    src/integrator/nve.cu
    src/integrator/grad_desc.cu
    src/particle/disk.cu
    src/particle/rigid_bumpy.cu
    src/integrator/adam.cu
    src/routines/initialization.cu
    src/io/utils.cu
    src/utils/thread_pool.cpp
)

# Link nlohmann_json to the common_sources library
target_link_libraries(common_sources PRIVATE nlohmann_json::nlohmann_json)

# Set properties for CUDA separable compilation
set_target_properties(common_sources PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

# Main executable
add_executable(main_exec scripts/main.cpp)
target_link_libraries(main_exec PRIVATE common_sources)

# Test executable
add_executable(test_exec scripts/test.cpp)
target_link_libraries(test_exec PRIVATE common_sources)

# Fast Jam executable
add_executable(fast_jam scripts/fast_jam.cpp)
target_link_libraries(fast_jam PRIVATE common_sources)

# Jam executable
add_executable(jam scripts/jam.cpp)
target_link_libraries(jam PRIVATE common_sources)

# Rigid Bumpy Jam executable
add_executable(rb_jam scripts/rb_jam.cpp)
target_link_libraries(rb_jam PRIVATE common_sources)

# Disk Pool executable
add_executable(disk_pool scripts/disk_pool.cpp)
target_link_libraries(disk_pool PRIVATE common_sources)

# Rigid Bumpy Pool executable
add_executable(rb_pool scripts/rb_pool.cpp)
target_link_libraries(rb_pool PRIVATE common_sources)

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
    ${CMAKE_SOURCE_DIR}/include/integrator
    ${CMAKE_SOURCE_DIR}/include/particle
    ${CMAKE_SOURCE_DIR}/include/kernels
    ${CMAKE_SOURCE_DIR}/include/io
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# Find the nlohmann_json package
find_package(nlohmann_json 3.2.0 REQUIRED)

# CC compiler flags (G++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fdevirtualize -MMD -MP")

# NVCC compiler flags (CUDA compiler)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-extended-lambda --expt-relaxed-constexpr -diag-suppress=550 -Wno-deprecated-gpu-targets -Xcompiler -fdevirtualize -MMD -MP")

# Libraries for C++ (e.g., linking stdc++fs for filesystem support)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")

# Add source files for the executable
add_executable(main_exec
    scripts/main.cpp
    src/kernels/kernels.cu
    src/particle/config.cu
    src/particle/particle.cu
    src/particle/particle_factory.cu
    src/io/orchestrator.cu
    src/particle/disk.cu
    src/io/utils.cu
)

# Link nlohmann_json to the main_exec target
target_link_libraries(main_exec PRIVATE nlohmann_json::nlohmann_json)

# # Add a new executable for another script
# add_executable(other_script_exec
#     scripts/other_script.cpp
#     src/integrator/integrator.cu
#     src/kernels/kernels.cu
#     src/particle/particle.cu
# )

# Set properties for CUDA separable compilation
set_target_properties(main_exec PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

# Add libraries or link with additional libraries if needed (optional)

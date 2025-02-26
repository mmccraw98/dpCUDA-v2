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
    ${CMAKE_SOURCE_DIR}/include/particles
    ${CMAKE_SOURCE_DIR}/include/particles/base
    ${CMAKE_SOURCE_DIR}/include/particles/disk
    ${CMAKE_SOURCE_DIR}/include/particles/rigid_bumpy
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
    src/particles/base/particle.cu
    src/particles/base/kernels.cu
    src/io/orchestrator.cu
    src/io/io_manager.cu
    src/io/base_log_groups.cu
    src/io/console_log.cu
    src/io/energy_log.cu
    src/io/state_log.cu
    src/integrator/integrator.cu
    src/integrator/nve.cu
    src/integrator/damped_nve.cu
    src/integrator/grad_desc.cu
    src/particles/disk/disk.cu
    src/particles/disk/kernels.cu
    src/particles/rigid_bumpy/rigid_bumpy.cu
    src/particles/rigid_bumpy/kernels.cu
    src/integrator/adam.cu
    src/routines/initialization.cu
    src/routines/compression.cu
    src/routines/minimization.cu
    src/io/io_utils.cu
    src/utils/thread_pool.cpp
    src/routines/quench.cu
)

# Link nlohmann_json to the common_sources library
target_link_libraries(common_sources PRIVATE nlohmann_json::nlohmann_json)

# Set properties for CUDA separable compilation
set_target_properties(common_sources PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

# # Test Load 2 executable
# add_executable(test_load_2 scripts/test_load_2.cpp)
# target_link_libraries(test_load_2 PRIVATE common_sources)

# # Jam executable
# add_executable(jam scripts/jam.cpp)
# target_link_libraries(jam PRIVATE common_sources)

# # Run NVE executable
# add_executable(run_nve scripts/run_nve.cpp)
# target_link_libraries(run_nve PRIVATE common_sources)

# # Test executable
# add_executable(test scripts/test.cpp)
# target_link_libraries(test PRIVATE common_sources)

# # Test create rigid bumpy executable
# add_executable(test_create_rigid_bumpy scripts/test_create_rigid_bumpy.cpp)
# target_link_libraries(test_create_rigid_bumpy PRIVATE common_sources)

# # Test load rigid bumpy executable
# add_executable(test_load_rigid_bumpy scripts/test_load_rigid_bumpy.cpp)
# target_link_libraries(test_load_rigid_bumpy PRIVATE common_sources)

# # Test create disk executable
# add_executable(test_create_disk scripts/test_create_disk.cpp)
# target_link_libraries(test_create_disk PRIVATE common_sources)

# # Test load disk executable
# add_executable(test_load_disk scripts/test_load_disk.cpp)
# target_link_libraries(test_load_disk PRIVATE common_sources)

# Run nve executable
add_executable(runNVE scripts/runNVE.cpp)
target_link_libraries(runNVE PRIVATE common_sources)

# Run rescale nvt executable
add_executable(runRescaleNVT scripts/runRescaleNVT.cpp)
target_link_libraries(runRescaleNVT PRIVATE common_sources)

# Run create samples rescale nvt executable
add_executable(runCreateSamplesRescaleNVT scripts/runCreateSamplesRescaleNVT.cpp)
target_link_libraries(runCreateSamplesRescaleNVT PRIVATE common_sources)

# Run damped jam executable
add_executable(runDampedJam scripts/runDampedJam.cpp)
target_link_libraries(runDampedJam PRIVATE common_sources)

# Run quench nve executable
add_executable(runQuenchNVE scripts/runQuenchNVE.cpp)
target_link_libraries(runQuenchNVE PRIVATE common_sources)

# Run post proc calc executable
add_executable(postProcCalc scripts/postProcCalc.cpp)
target_link_libraries(postProcCalc PRIVATE common_sources)

# Run icp pack executable
add_executable(runICPPack scripts/runICPPack.cpp)
target_link_libraries(runICPPack PRIVATE common_sources)

# Run Arthur jam executable
add_executable(runArthurJam scripts/runArthurJam.cpp)
target_link_libraries(runArthurJam PRIVATE common_sources)

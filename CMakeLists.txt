cmake_minimum_required(VERSION 3.10)
project(CUDA_Matrix_Multiplication CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

add_executable(CUDA_Matrix_Multiplication cumatrix.cu main.cpp)

set_target_properties(CUDA_Matrix_Multiplication PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SPARABLE_COMPILATION ON)

target_link_libraries(CUDA_Matrix_Multiplication -lcublas)
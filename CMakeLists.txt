cmake_minimum_required(VERSION 3.16)

# CCACHE
find_program(CCACHE "ccache")
if(CCACHE)
    message("CCACHE FOUND! = ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif(CCACHE)

project(CeresIcp)

# Configure ...
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# set(CMAKE_CUDA_ARCHITECTURES 62)

# CUDA config ...
include(CheckLanguage)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
enable_language(CUDA)
check_language(CUDA)

# add_definitions(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache)

# HMM...
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 )

include(cmake/FindEigen3.cmake)
# set(Eigen3_DIR /usr/local/share/eigen3)
# set(CUDAToolkit_ROOT "/usr/local/cuda")
find_package(Eigen3 3.3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Open3D REQUIRED)
find_package(fmt REQUIRED)
# find_package(CUDAToolkit REQUIRED)


function(dump_cmake_variables)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        if (ARGV0)
            unset(MATCHED)
            string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
            if (NOT MATCHED)
                continue()
            endif()
        endif()
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

add_subdirectory(ceres_icp)
# dump_cmake_variables()

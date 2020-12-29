cmake_minimum_required(VERSION 3.16)
project(CeresIcp)

# Configure ...
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(cmake/FindEigen3.cmake)
set(Eigen3_DIR /usr/local/share/eigen3)

#find_package(Eigen 3.3 REQUIRED)
#include_directories(${eigen3_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
find_package(Open3D REQUIRED)
find_package(fmt REQUIRED)

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

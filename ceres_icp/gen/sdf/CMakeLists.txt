add_library(cho_gen_sdf SHARED
    src/ray_marching.cpp
    include/cho/gen/sdf_fwd.hpp
    include/cho/gen/sdf.hpp
    include/cho/gen/ray_marching.hpp
    )
target_link_libraries(cho_gen_sdf Eigen3::Eigen)

target_include_directories(cho_gen_sdf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_subdirectory(test)

add_library(cho_gen_sdf SHARED
    src/ray_marching.cpp
    include/cho/gen/sdf_fwd.hpp
    include/cho/gen/sdf_types.hpp
    include/cho/gen/sdf.hpp
    src/sdf.cpp

    include/cho/gen/ray_marching.hpp

    include/cho/gen/gen_utils.hpp
    src/gen_utils.cpp

    include/cho/gen/sdf_utils.hpp
    src/sdf_utils.cpp
    )
target_link_libraries(cho_gen_sdf Eigen3::Eigen fmt::fmt)

target_include_directories(cho_gen_sdf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_subdirectory(cuda)
add_subdirectory(test)

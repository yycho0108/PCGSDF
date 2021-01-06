add_library(sdf_render_cu SHARED
    src/render.cu
    include/cho/gen/cuda/render.hpp
    include/cho/gen/cuda/common.hpp
    include/cho/gen/cuda/cutil_math.cuh
    )
target_compile_options(sdf_render_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    >)
target_link_libraries(sdf_render_cu PRIVATE Eigen3::Eigen cho_gen_sdf)
target_include_directories(sdf_render_cu PUBLIC include/)
set_target_properties(sdf_render_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 50
    )
# set_property(TARGET test_ray_group PROPERTY CUDA_ARCHITECTURES 50)
# add_subdirectory(jit)
add_subdirectory(test)

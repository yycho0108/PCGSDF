

add_library(sdf_render_cu SHARED
    src/render.cu
    include/cho/gen/cuda/render.hpp
    include/cho/gen/cuda/common.hpp
    include/cho/gen/cuda/cutil_math.cuh
    )
target_compile_options(sdf_render_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    >)
target_link_libraries(sdf_render_cu PRIVATE Eigen3::Eigen cho_gen_sdf)
target_include_directories(sdf_render_cu PUBLIC include/)
set_target_properties(sdf_render_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# just in time!
find_package(CUDAToolkit REQUIRED)

add_library(sdf_render_cu_jit SHARED
    src/render_jit.cu
    include/cho/gen/cuda/render_jit.hpp
    include/cho/gen/cuda/cutil_math.cuh
    )
target_compile_options(sdf_render_cu_jit PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    >)
target_link_libraries(sdf_render_cu_jit PRIVATE Eigen3::Eigen fmt::fmt CUDA::nvrtc)
target_include_directories(sdf_render_cu_jit PUBLIC include/)
set_target_properties(sdf_render_cu_jit PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

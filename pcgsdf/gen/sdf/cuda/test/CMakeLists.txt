add_executable(test_ray_group test_ray_group.cu)
# target_link_libraries(cho_gen_sdf_test PRIVATE cho_gen_sdf fmt::fmt ${Open3D_LIBRARIES})
# target_link_libraries(cho_gen_sdf_test PRIVATE sdf_render_cu_jit)
# target_link_libraries(cho_gen_sdf_test PRIVATE sdf_render_cu)
# target_include_directories(cho_gen_sdf_test PRIVATE ${Open3D_INCLUDE_DIRS})
# target_compile_definitions(cho_gen_sdf_test PRIVATE _GLIBCXX_USE_CXX11_ABI=1) 
set_target_properties(test_ray_group PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_ray_group PROPERTY CUDA_ARCHITECTURES 50)


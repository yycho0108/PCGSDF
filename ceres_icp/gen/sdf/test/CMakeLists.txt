add_executable(cho_gen_sdf_test
    test_sdf.cpp)
target_link_libraries(cho_gen_sdf_test PRIVATE cho_gen_sdf sdf_render_cu fmt::fmt ${Open3D_LIBRARIES})
target_include_directories(cho_gen_sdf_test PRIVATE ${Open3D_INCLUDE_DIRS})

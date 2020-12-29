#add_executable(main main.cpp)
#target_link_libraries(main PRIVATE ${Ceres_LIBRARIES})
#target_include_directories(main PRIVATE ${Ceres_INCLUDE_DIRS}
#    ${eigen3_INCLUDE_DIRS})

add_executable(vis vis.cpp)
target_link_libraries(vis PRIVATE ${Ceres_LIBRARIES} ${Open3D_LIBRARIES} fmt::fmt)
target_include_directories(vis PRIVATE ${Ceres_INCLUDE_DIRS}
    ${eigen3_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} )

add_subdirectory(gen)
# dump_cmake_variables()

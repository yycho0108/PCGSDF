#add_executable(main main.cpp)
#target_link_libraries(main PRIVATE ${Ceres_LIBRARIES})
#target_include_directories(main PRIVATE ${Ceres_INCLUDE_DIRS}
#    ${eigen3_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)

#add_executable(vis vis.cpp)
#target_link_libraries(vis PRIVATE ${Ceres_LIBRARIES} ${Open3D_LIBRARIES} fmt::fmt Eigen3::Eigen)
#target_include_directories(vis PRIVATE ${Ceres_INCLUDE_DIRS}
#    ${eigen3_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} )

add_subdirectory(gen)
add_subdirectory(vis)
# dump_cmake_variables()

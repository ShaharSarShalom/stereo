# Generator
halide_project(guidedFilter "StereoHalide" guidedFilter.cpp)

set(guidedFilter_h "${CMAKE_CURRENT_BINARY_DIR}/guidedFilter.h")
set(guidedFilter_obj "${CMAKE_CURRENT_BINARY_DIR}/guidedFilter${CMAKE_CXX_OUTPUT_EXTENSION}")

# Final executable

add_executable(runGF runGF.cpp ${guidedFilter_h})
target_link_libraries(runGF PRIVATE HALIDE ${guidedFilter_obj} ${PNG_LIBRARIES} ${IMAGE_LIB})
target_include_directories(runGF PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
                           "${PNG_INCLUDE_DIRS}" "${IMAGE_LIB_DIR}")

if (NOT WIN32)
  target_link_libraries(runGF PRIVATE dl pthread)
endif()

if (NOT MSVC)
  target_compile_options(runGF PRIVATE "-std=c++11")
endif()

# FIXME: Cannot use halide_add_generator_dependency() because
# guidedFilter doesn't handle the commandline args passed.
add_custom_command(OUTPUT "${guidedFilter_h}" "${guidedFilter_obj}"
                   COMMAND guidedFilter
                   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                   COMMENT "Generating guidedFilter"
                  )

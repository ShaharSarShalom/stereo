# project
cmake_minimum_required (VERSION 2.8)
project (StereoHalide)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
set(Boost_USE_STATIC_LIBS ON)
set(HALIDE_DIR "/Users/mzhang94/Halide")
include_directories("${HALIDE_DIR}/include")
include_directories("${HALIDE_DIR}/tools")
include_directories("${CMAKE_SOURCE_DIR}/include")

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

find_package( OpenCV REQUIRED )

SET(IMAGE_LIB "~/MiddEval3/code/imageLib")
link_directories( ${IMAGE_LIB} )
include_directories( ${IMAGE_LIB} )
include_directories(lib)

file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( Stereo ${LIB_SOURCES} ${LIB_HEADERS} )

# make release version
set(CMAKE_BUILD_TYPE Release)

# build executable
file( GLOB APP_SOURCES src/*.cxx )
macro(add_test testname source_file)
    add_executable (${testname} ${source_file})
    target_link_libraries(  ${testname}
                            ${PNG_LIBRARY}
                            "Img.i386-g"
                            ${HALIDE_DIR}/bin/libHalide.so
                            Stereo
                            ${OpenCV_LIBS}
                          )
endmacro()

add_test(runBM src/runBM.cpp)
add_test(runGF src/runGF.cpp)
add_test(test src/test.cpp)

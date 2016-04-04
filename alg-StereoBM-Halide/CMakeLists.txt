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

# include directory
include_directories(src)

# sources
FILE(GLOB SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

# build executable
add_executable(stereoBM_halide ${SRC_FILES})
target_link_libraries( stereoBM_halide
    ${PNG_LIBRARY}
    "Img.i386-g"
    ${HALIDE_DIR}/bin/libHalide.so
    ${OpenCV_LIBS})

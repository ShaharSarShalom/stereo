# project
cmake_minimum_required (VERSION 2.6)
project (StereoOpenCV)
find_package( OpenCV REQUIRED )

find_package( PNG REQUIRED )
include_directories(${PNG_INCLUDE_DIR})

SET(IMAGE_LIB "~/MiddEval3/code/imageLib")
link_directories( ${IMAGE_LIB} )
include_directories( ${IMAGE_LIB} )

# include directory
include_directories("src")

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)

# build demo program
add_executable(StereoOpenCV ${SRC_FILES})
target_link_libraries( StereoOpenCV "Img.i386-g" ${PNG_LIBRARY} ${OpenCV_LIBS} )

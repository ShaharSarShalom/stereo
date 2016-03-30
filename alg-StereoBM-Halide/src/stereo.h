#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

typedef unsigned short ushort;
Func stereoBM(Image<int8_t> left_image, Image<int8_t> right_image, int SADWindowSize, int minDisparity, int numDisparities);

#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

extern short FILTERED;
typedef unsigned short ushort;
float profile(Func myFunc, int w, int h);
Image<short> stereoBM(Image<int8_t> left_image, Image<int8_t> right_image, int SADWindowSize, int minDisparity,
              int numDisparities, int xmin, int xmax, int ymin, int ymax);

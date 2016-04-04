#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

extern short FILTERED;
typedef unsigned short ushort;
float profile(Func myFunc, int w, int h);
Image<ushort> stereoBM(Image<uint8_t> left_image, Image<uint8_t> right_image, int SADWindowSize, int minDisparity,
              int numDisparities, int xmin, int xmax, int ymin, int ymax);

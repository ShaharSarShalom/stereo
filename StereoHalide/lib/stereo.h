#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

extern short FILTERED;
typedef unsigned short ushort;
float profile(Func myFunc, int w, int h, int n_times);
float profile(Func myFunc, int w, int h, int c, int n_times);
float profile(Func myFunc, Buffer buf, int n_times);
Image<ushort> stereoBM(Image<uint8_t> left_image, Image<uint8_t> right_image, int SADWindowSize, int minDisparity,
              int numDisparities, int xmin, int xmax, int ymin, int ymax, bool useGPU);
Func guidedFilter_gray(Func I, Func p, int r, float epsilon);
Func guidedFilter(Func I, Func p, int r, float epsilon);
Func stereoGF(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad);
Func stereoGF_scheduled(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad);
Func stereoGF_fast(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad);

void guidedFilterTest();

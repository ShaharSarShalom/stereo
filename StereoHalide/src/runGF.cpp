#include <unistd.h>
#include <sys/time.h>
#include "stereo.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "imageLib.h"

using namespace Halide;
using namespace Halide::Tools;

template <class T>
CFloatImage convertHalideImageToFloatImage(Image<T> image, int xmin, int xmax, int ymin, int ymax) {
    CFloatImage img(image.width(), image.height(), 1);
    for (int x = 0; x < image.width(); x++) {
        for (int y = 0; y < image.height(); y++) {
            float* ptr = (float *) img.PixelAddress(x, y, 0);
            short pixel_val = image(x-xmin, y-ymin);
            if (pixel_val < 0 || x < xmin || x > xmax || y < ymin || y > ymax)
            {
                *ptr = INFINITY;
            }
            else
            {
                *ptr = (float)pixel_val;
            }
        }
    }
    return img;
}

int main(int argc, char** argv ) {

    if (argc < 4)
    {
        printf("Need to provide name of left right images, output filename and disparity range\n");
    }
    const char* im0_name = argv[1];
    const char* im1_name = argv[2];
    const char* disp_image_name = argv[3];

    Image<float> im0 = load_image(im0_name);
    Image<float> im1 = load_image(im1_name);

    int numDisparities = 0;
    if( sscanf( argv[4], "%d", &numDisparities ) != 1 || numDisparities < 1)
    {
        printf("Disparity range invalid\n");
    }


    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    right(x, y, c) = im1(clamp(x, 0, im1.width()-1), clamp(y, 0, im1.height()-1), c);

    Func disp = stereoGF_fast(left, right, im0.width(), im0.height(), 9, 0.0001, numDisparities, 0.9, 0.0028, 0.008);
    Image<int> disp_image = disp.realize(im0.width(), im1.height());
    Image<float> scaled_disp(disp_image.width(), disp_image.height());

    int maxDisparity = numDisparities - 1;
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = std::min(1.f, std::max(0.f, disp_image(x,y) * 1.0f / maxDisparity));
        }
    };
    profile(disp, im0.width(), im0.height(), 10);
    save_image(scaled_disp, "disp.png");
    WriteFilePFM(convertHalideImageToFloatImage<int>(disp_image, 0, im0.width(), 0, im0.height()), disp_image_name, 1.0f/maxDisparity);
}

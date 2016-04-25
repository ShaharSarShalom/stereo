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
    // Image<float> image = load_image("../../trainingQ/Teddy/im0.png");
    //
    // Var x("x"), y("y"), c("c");
    // Func input("input");
    //
    // Func t("t");
    // t(x, y, c) = 0.5f;
    //
    // input(x, y, c) = image(clamp(x, 0, image.width()-1), clamp(y, 0, image.height()-1), c);
    // Func filtered = guidedFilter(input, input, 4, 0.01);
    // Func clamped("clamp");
    // clamped(x, y, c) = clamp(filtered(x, y, c), 0, 1);
    //
    // Image<float> output = clamped.realize(image.width(), image.height(), 3);
    // save_image(output, "filtered.png");
    // std::cout << output(0,0,0);
    // Func enhanced("enhanced");
    // enhanced(x, y, c) = clamp((input(x, y, c) - filtered(x, y, c))*5 + input(x, y, c), 0, 1);
    // output = enhanced.realize(image.width(), image.height(), 3);
    // save_image(output, "enhanced.png");

    Image<float> im0 = load_image("../../trainingQ/Teddy/im0.png");
    Image<float> im1 = load_image("../../trainingQ/Teddy/im1.png");
    int numDisparities = 60;

    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    right(x, y, c) = im1(clamp(x, 0, im1.width()-1), clamp(y, 0, im1.height()-1), c);

    // Func gray_left("gray_left"), gray_right("gray_right");
    // gray_left(x, y) = 0.2989f*left(x, y, 0) + 0.5870f*left(x, y, 1) + 0.1140f*left(x, y, 2);
    // gray_right(x, y) = 0.2989f*right(x, y, 0) + 0.5870f*right(x, y, 1) + 0.1140f*right(x, y, 2);

    Func disp = stereoGF(left, right, im0.width(), im0.height(), 9, 0.0001, numDisparities, 0.9, 7/255.0, 2/255.0);
    Image<int> disp_image = disp.realize(im0.width(), im1.height());
    Image<float> scaled_disp(disp_image.width(), disp_image.height());

    int maxDisparity = numDisparities - 1;
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = std::min(1.f, std::max(0.f, disp_image(x,y) * 1.0f / maxDisparity));
        }
    };
    printf("%f\n", INFINITY);
    save_image(scaled_disp, "disp.png");
    WriteFilePFM(convertHalideImageToFloatImage<int>(disp_image, 0, im0.width(), 0, im0.height()), "../../trainingQ/Teddy/disp0GF-Halide.pfm", 1.0f/maxDisparity);
}

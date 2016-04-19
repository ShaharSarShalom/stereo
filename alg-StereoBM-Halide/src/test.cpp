#include <unistd.h>
#include <sys/time.h>
#include "stereo.h"
#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;
int main(int argc, char** argv ) {
    // Image<float> image = load_image("../gray_small.png");
    //
    // Var x("x"), y("y");
    // Func input("input");
    //
    // Func t("t");
    // t(x, y) = 1.0f;
    //
    // input(x, y) = image(clamp(x, 0, image.width()-1), clamp(y, 0, image.height()-1));
    // Func filtered = guidedFilter_gray(input, input, 4, 0.01);
    // Func clamped("clamp");
    // clamped(x, y) = clamp(filtered(x, y), 0, 1);
    //
    // Image<float> output = clamped.realize(image.width(), image.height());
    // profile(clamped, image.width(), image.height(), 1000);
    // save_image(output, "output.png");

    Image<float> im0 = load_image("../../trainingQ/Teddy/im0.png");
    Image<float> im1 = load_image("../../trainingQ/Teddy/im1.png");
    int numDisparities = 60;

    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    right(x, y, c) = im1(clamp(x, 0, im1.width()-1), clamp(y, 0, im1.height()-1), c);

    Func gray_left("gray_left"), gray_right("gray_right");
    gray_left(x, y) = 0.2989f*left(x, y, 0) + 0.5870f*left(x, y, 1) + 0.1140f*left(x, y, 2);
    gray_right(x, y) = 0.2989f*right(x, y, 0) + 0.5870f*right(x, y, 1) + 0.1140f*right(x, y, 2);

    Func disp = stereoGF(gray_left, gray_right, im0.width(), im0.height(), 9, 0.0001, numDisparities, 0.9, 7/255.0, 2/255.0);
    Image<int> disp_image = disp.realize(im0.width(), im1.height());
    Image<float> scaled_disp(disp_image.width(), disp_image.height());

    int maxDisparity = numDisparities - 1;
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = std::min(1.f, std::max(0.f, disp_image(x,y) * 1.0f / maxDisparity));
        }
    };
    save_image(scaled_disp, "disp.png");

}

#include <unistd.h>
#include <sys/time.h>
#include "stereo.h"
#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;
int main(int argc, char** argv ) {
    Image<float> image = load_image("../../trainingQ/Adirondack/im0.png");

    Var x("x"), y("y"), c("c");
    Func input("input");

    Func t("t");
    t(x, y, c) = 0.5f;

    input(x, y, c) = image(clamp(x, 0, image.width()-1), clamp(y, 0, image.height()-1), c);
    Func filtered = guidedFilter(input, input, 9, 0.01);
    Func clamped("clamp");
    clamped(x, y, c) = clamp(filtered(x, y, c), 0, 1);

    Image<float> output = clamped.realize(image.width(), image.height(), 3);
    save_image(output, "filtered.png");
    Func enhanced("enhanced");
    enhanced(x, y, c) = clamp((input(x, y, c) - filtered(x, y, c))*5 + input(x, y, c), 0, 1);
    output = enhanced.realize(image.width(), image.height(), 3);
    std::cout << output(100,100,0);
    save_image(output, "enhanced.png");
}

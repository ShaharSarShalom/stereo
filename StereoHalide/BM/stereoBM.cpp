#include "Halide.h"
#include "halide_image_io.h"
#include "stereo.h"
#include <limits>

using namespace Halide;
short FILTERED = -16;

Func prefilterXSobel(Func image, int w, int h) {
    Var x("x"), y("y");
    Func clamped("clamped"), gray("gray");
    gray(x, y) = 0.2989f*image(x, y, 0) + 0.5870f*image(x, y, 1) + 0.1140f*image(x, y, 2);
    clamped(x, y) = gray(clamp(x, 0, w-1), clamp(y, 0, h-1));

    Func temp("temp"), xSobel("xSobel");
    temp(x, y) = clamped(x+1, y) - clamped(x-1, y);
    xSobel(x, y) = cast<short>(clamp(temp(x, y-1) + 2 * temp(x, y) + temp(x, y+1), -31, 31));

    // Schedule
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    xSobel.compute_root().tile(x, y, xo, yo, xi, yi, 64, 32).parallel(yo).parallel(xo);
    temp.compute_at(xSobel, yi).vectorize(x, 8);
    return xSobel;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int width, int height, int xmin, int xmax, int ymin, int ymax,
    int x_tile_size = 32, int y_tile_size = 32, bool test = false, float uniquenessRatio = 0.15, int disp12MaxDiff = 1) {

    Var x("x"), y("y"), c("c"), d("d");

    Func diff("diff");
    diff(d, x, y) = cast<ushort>(abs(left(x, y) - right(x-d, y)));

    int win2 = SADWindowSize/2;

    Func diff_T("diff_T");
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    diff_T(d, xi, yi, xo, yo) = diff(d, xi+xo*x_tile_size+xmin, yi+yo*y_tile_size+ymin);

    Func cSAD("cSAD"), vsum("vsum");
    RDom rk(-win2, SADWindowSize, "rk");
    RDom rxi(1, x_tile_size - 1, "rxi"), ryi(1, y_tile_size - 1, "ryi");
    if (test) {
        vsum(d, xi, yi, xo, yo) = sum(diff_T(d, xi, yi+rk, xo, yo));
        cSAD(d, xi, yi, xo, yo) = sum(vsum(d, xi+rk, yi, xo, yo));
    }
    else{
        vsum(d, xi, yi, xo, yo) = select(yi != 0, cast<ushort>(0), sum(diff_T(d, xi, rk, xo, yo)));
        vsum(d, xi, ryi, xo, yo) = vsum(d, xi, ryi-1, xo, yo) + diff_T(d, xi, ryi+win2, xo, yo) - diff_T(d, xi, ryi-win2-1, xo, yo);

        cSAD(d, xi, yi, xo, yo) = sum(vsum(d, xi+rk, yi, xo, yo));
    }

    RDom rd(minDisparity, numDisparities);
    Func disp_left("disp_left");
    disp_left(xi, yi, xo, yo) = {cast<ushort>(minDisparity), cast<ushort>((2<<16)-1)};
    disp_left(xi, yi, xo, yo) = tuple_select(
            cSAD(rd, xi, yi, xo, yo) < disp_left(xi, yi, xo, yo)[1],
            {cast<ushort>(rd), cSAD(rd, xi, yi, xo, yo)},
            disp_left(xi, yi, xo, yo));

    Func disp("disp");
    disp(x, y) = select(
        x>xmax-xmin || y>ymax-ymin,
        cast<ushort>(FILTERED),
        cast<ushort>(disp_left(x%x_tile_size, y%y_tile_size, x/x_tile_size, y/y_tile_size)[0]));

    int vector_width = 8;

    // Schedule
    disp.compute_root().tile(x, y, xo, yo, xi, yi, x_tile_size, y_tile_size).reorder(xi, yi, xo, yo)
        .vectorize(xi, vector_width).parallel(xo).parallel(yo);

    // reorder storage
    disp_left.reorder_storage(xi, yi, xo, yo);
    diff_T   .reorder_storage(xi, yi, xo, yo, d);
    vsum     .reorder_storage(xi, yi, xo, yo, d);
    cSAD     .reorder_storage(xi, yi, xo, yo, d);

    disp_left.compute_at(disp, xo).reorder(xi, yi, xo, yo)    .vectorize(xi, vector_width)
             .update()            .reorder(xi, yi, rd, xo, yo).vectorize(xi, vector_width);

    if (test){
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
    }
    else {
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
            .update()                 .reorder(xi, ryi, xo, yo, d).vectorize(xi, vector_width);
    }

    return disp;
}


Image<ushort> stereoBM(Image<uint8_t> left_image, Image<uint8_t> right_image, int SADWindowSize, int minDisparity,
              int numDisparities, int xmin, int xmax, int ymin, int ymax) {
    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = left_image(x, y, c);
    right(x, y, c) = right_image(x, y, c);

    int width = left_image.width();
    int height = left_image.height();

    Func filteredLeft = prefilterXSobel(left, width, height);
    Func filteredRight = prefilterXSobel(right, width, height);

    int x_tile_size = 64, y_tile_size = 32;
    Func disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
        left_image.width(), left_image.height(), xmin, xmax, ymin, ymax, x_tile_size, y_tile_size);
    disp.compile_to_lowered_stmt("disp.html", {}, HTML);

    int w = (xmax-xmin)/x_tile_size*x_tile_size+x_tile_size;
    int h = (ymax-ymin)/x_tile_size*x_tile_size+x_tile_size;

    profile(disp, w, h, 100);
    Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    Image<ushort> disp_image = disp.realize(w, h, t);

    return disp_image;
}

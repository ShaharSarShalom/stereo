#include "Halide.h"
#include "halide_image_io.h"
#include "stereo.h"
#include <limits>

using namespace Halide;
short FILTERED = -16;

Func prefilterXSobel(Func image, int w, int h) {
    Var x("x"), y("y");
    Func clamped("clamped"), gray("gray");
    gray(x, y) = cast<int8_t>(0.2989f*image(x, y, 0) + 0.5870f*image(x, y, 1) + 0.1140f*image(x, y, 2));
    clamped(x, y) = gray(clamp(x, 0, w-1), clamp(y, 0, h-1));

    Func temp("temp"), xSobel("xSobel");
    temp(x, y) = clamped(x+1, y) - clamped(x-1, y);
    xSobel(x, y) = cast<short>(clamp(temp(x, y-1) + 2 * temp(x, y) + temp(x, y+1), -31, 31) + 31);

    // Schedule
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    xSobel.compute_root().tile(x, y, xo, yo, xi, yi, 64, 32).parallel(yo).parallel(xo);
    temp.compute_at(xSobel, yi).vectorize(x, 8);
    return xSobel;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int width, int height, bool test = false, float uniquenessRatio = 0.15, int disp12MaxDiff = 1) {

    Var x("x"), y("y"), c("c"), d("d");

    Func diff("diff");
    diff(d, x, y) = cast<ushort>(abs(left(x, y) - right(x-d, y)));

    int win2 = SADWindowSize/2;
    int minD = minDisparity, maxD = minDisparity + numDisparities - 1;
    int xmin = maxD + win2;
    int xmax = width - minD - win2;
    int ymin = win2;
    int ymax = height - win2;

    int x_tile_size = 32, y_tile_size = 32;

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

        cSAD(d, xi, yi, xo, yo) = select(xi != 0, cast<ushort>(0), sum(vsum(d, rk, yi, xo, yo)));
        cSAD(d, rxi, yi, xo, yo) = cSAD(d, rxi-1, yi, xo, yo) + vsum(d, rxi+win2, yi, xo, yo) - vsum(d, rxi-win2-1, yi, xo, yo);
    }

    RDom rd(minDisparity, numDisparities);
    Func disp_left("disp_left");
    disp_left(xi, yi, xo, yo) = {cast<ushort>(minDisparity), cast<ushort>((2<<16)-1)};
    disp_left(xi, yi, xo, yo) = tuple_select(
            cSAD(rd, xi, yi, xo, yo) < disp_left(xi, yi, xo, yo)[1],
            {cast<ushort>(rd), cSAD(rd, xi, yi, xo, yo)},
            disp_left(xi, yi, xo, yo));

    Func disp("disp");
    disp(x, y) = cast<short>(FILTERED); // undef means this line wont be executed, use this if you dont need to initialize
    int num_x_tiles = (xmax-xmin) / x_tile_size, num_y_tiles = (ymax-ymin) / y_tile_size;
    RDom rr(0, x_tile_size, 0, y_tile_size, 0, num_x_tiles, 0, num_y_tiles);
    Expr rx = rr[0] + rr[2] * x_tile_size + xmin;
    Expr ry = rr[1] + rr[3] * y_tile_size + ymin;

    disp(rx, ry) = select(
                    rx > xmax || ry >= ymax,
                    FILTERED,
                    cast<short>(disp_left(rr[0], rr[1], rr[2], rr[3])[0])
                 );

    int vector_width = 8;

    // Schedule
    disp.compute_root().parallel(y, 8).vectorize(x, vector_width)
        .update().unroll(rr[0]);

    // reorder storage
    disp_left.reorder_storage(xi, yi, xo, yo);
    diff_T   .reorder_storage(xi, yi, xo, yo, d);
    vsum     .reorder_storage(xi, yi, xo, yo, d);
    cSAD     .reorder_storage(xi, yi, xo, yo, d);

    disp_left.compute_root().reorder(xi, yi, xo, yo)    .vectorize(xi, vector_width).parallel(xo).parallel(yo)
             .update()      .reorder(xi, yi, rd, xo, yo).vectorize(xi, vector_width).parallel(xo).parallel(yo);

    if (test){
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
    }
    else {
        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
            .update()                 .reorder(yi, rxi, xo, yo, d).vectorize(yi, vector_width);
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
            .update()                 .reorder(xi, ryi, xo, yo, d).vectorize(xi, vector_width);
    }


    return disp;
}


Func stereoBM(Image<int8_t> left_image, Image<int8_t> right_image, int SADWindowSize, int minDisparity, int numDisparities) {
    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = left_image(x, y, c);
    right(x, y, c) = right_image(x, y, c);

    int width = left_image.width();
    int height = left_image.height();

    Func filteredLeft = prefilterXSobel(left, width, height);
    Func filteredRight = prefilterXSobel(right, width, height);

    /* get valid disparity region */
    Func disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
        left_image.width(), left_image.height());

    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    return disp;
}

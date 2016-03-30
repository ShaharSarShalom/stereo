#include "Halide.h"
#include "halide_image_io.h"
#include <limits>

using namespace Halide;

int FILTERED = -16;

Func prefilterXSobel(Func image, int w, int h) {
    Var x("x"), y("y");
    Func clamped("clamped"), gray("gray");
    gray(x, y) = cast<int8_t>(0.2989f*image(x, y, 0) + 0.5870f*image(x, y, 1) + 0.1140f*image(x, y, 2));
    clamped(x, y) = gray(clamp(x, 0, w-1), clamp(y, 0, h-1));

    Func temp("temp"), xSobel("xSobel");
    temp(x, y) = clamped(x+1, y) - clamped(x-1, y);
    xSobel(x, y) = cast<short>(clamp(temp(x, y-1) + 2 * clamped(x, y) + clamped(x, y+1), -31, 31) + 31);

    // Schedule
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    xSobel.compute_root();
    xSobel.tile(x, y, xo, yo, xi, yi, 128, 32);
    temp.compute_at(xSobel, yi).vectorize(x, 16);
    xSobel.parallel(yo);
    return xSobel;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int width, int height, float uniquenessRatio = 0.15, int disp12MaxDiff = 1) {

    Var x("x"), y("y"), c("c"), d("d");

    Func diff("diff");
    diff(d, x, y) = cast<ushort>(abs(left(x, y) - right(x-d, y)));

    int win2 = SADWindowSize/2;
    int minD = minDisparity, maxD = minDisparity + numDisparities - 1;
    int xmin = maxD + win2;
    int xmax = width - minD - win2;
    int ymin = win2;
    int ymax = height - win2;

    int x_tile_size = 64, y_tile_size = 32;

    Func diff_T("diff_T");
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    diff_T(d, xi, yi, xo, yo) = diff(d, xi+xo*x_tile_size+xmin, yi+yo*y_tile_size+ymin);

    Func cSAD("cSAD");
    cSAD(d, xi, yi, xo, yo) = {(ushort)0, (ushort)0};
    RDom r(-win2, x_tile_size + win2, -win2, y_tile_size + win2, "r");
    RVar rxi = r.x, ryi = r.y;
    Expr new_vsum = cSAD(d, rxi, ryi-1, xo, yo)[0] + diff_T(d, rxi+win2, ryi+win2, xo, yo) - select(ryi <= win2, 0, diff_T(d, rxi+win2, ryi-win2-1, xo, yo));
    Expr new_sum = cSAD(d, rxi-1, ryi, xo, yo)[1] + new_vsum - select(rxi <= win2, 0, cSAD(d, rxi-SADWindowSize, ryi, xo, yo)[0]);
    cSAD(d, rxi, ryi, xo, yo) = {new_vsum, new_sum};

    int vec_size = 16, num_vecs = numDisparities/vec_size;
    RDom k(0, num_vecs, "k"); Var j("j");
    Func disp_left("disp_left"), vec_disp_left("vec_disp_left");
    vec_disp_left(j, xi, yi, xo, yo) = {(ushort)minDisparity, (ushort)((2<<16)-1)};
    Expr rd = k * vec_size + j + minDisparity;
    vec_disp_left(j, xi, yi, xo, yo) = tuple_select(
                            cSAD(rd, xi, yi, xo, yo)[1] < vec_disp_left(j, xi, yi, xo, yo)[1],
                            {rd, cSAD(rd, xi, yi, xo, yo)[1]},
                            vec_disp_left(j, xi, yi, xo, yo));
    disp_left(xi, yi, xo, yo) = {(ushort)minDisparity, (ushort)((2>>16)-1)};
    RDom rj(0, vec_size, "rj");
    disp_left(xi, yi, xo, yo) = tuple_select(
                            vec_disp_left(rj, xi, yi, xo, yo)[1] < disp_left(xi, yi, xo, yo)[1],
                            vec_disp_left(rj, xi, yi, xo, yo),
                            disp_left(xi, yi, xo, yo));
    // check unique match
    // Func unique("unique");
    // unique(x, y) = 1;
    // unique(x, y) = select(rd != disp_left(x, y)[0] && cSAD(rd, x, y)[1] <= disp_left(x, y)[1] * (1 + uniquenessRatio), 0, 1);

    // validate disparity by comparing left and right
    // calculate disp2
    // Func disp_right("disp_right");
    // RDom rx1 = RDom(xmin, xmax - xmin + 1);
    // disp_right(x, y) = {minDisparity, INT_MAX};
    // Expr x2 = clamp(rx1 - disp_left(rx1, y)[0], xmin, xmax);
    // disp_right(x2, y) = tuple_select(
    //                          disp_right(x2, y)[1] > disp_left(rx1, y)[1],
    //                          disp_left(rx1, y),
    //                          disp_right(x2, y)
    //                      );

    Func disp("disp");
    // Expr x2 = clamp(x - disp_left(xi, yi)[0], xmin, xmax);
    // disp(x, y) = select(
    //                 unique(x, y) == 0
    //                 || !(x >= xmin && x <= xmax && y >= ymin && y<= ymax)
    //                 || (x2 >= xmin && x2 <= xmax && abs(disp_right(x2, y)[0] - disp_left(x, y)[0]) > disp12MaxDiff),
    //                 FILTERED,
    //                 disp_left(x,y)[0]
    //              );

    disp(x, y) = FILTERED;
    int num_x_tiles = (xmax-xmin) / x_tile_size, num_y_tiles = (ymax-ymin) / y_tile_size;
    RDom rr(0, x_tile_size, 0, y_tile_size, 0, num_x_tiles, 0, num_y_tiles);
    Expr rx = rr[0] + rr[2] * x_tile_size + xmin;
    Expr ry = rr[1] + rr[3] * y_tile_size + ymin;

    disp(rx, ry) = select(
                    rx > xmax || ry >= ymax,
                    FILTERED,
                    disp_left(rr[0], rr[1], rr[2], rr[3])[0]
                 );

    // Schedule
    disp.compute_root();
    disp_left.update().unroll(rj);
    vec_disp_left.compute_at(disp_left, xi)
                 .update().vectorize(j);

    cSAD.compute_at(disp, rr[2]);
    cSAD.update().reorder(d, rxi, ryi, xo, yo).vectorize(d, 8);
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

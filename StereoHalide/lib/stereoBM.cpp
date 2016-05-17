#include "Halide.h"
#include "halide_image_io.h"
#include "stereo.h"
#include <limits>

using namespace Halide;
short FILTERED = -16;

Func prefilterXSobel(Func image, int w, int h, bool schedule_GPU) {
    Var x("x"), y("y");
    Func clamped("clamped");
    clamped(x, y) = image(clamp(x, 0, w-1), clamp(y, 0, h-1));

    Func temp("temp"), xSobel("xSobel");
    temp(x, y) = clamped(x+1, y) - clamped(x-1, y);
    xSobel(x, y) = cast<short>(clamp(temp(x, y-1) + 2 * temp(x, y) + temp(x, y+1), -31, 31));

    // Schedule
    if (!schedule_GPU)
    {
        Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
        xSobel.compute_root().tile(x, y, xo, yo, xi, yi, 64, 32).vectorize(xi, 8).parallel(yo).parallel(xo);
        temp.compute_at(xSobel, yi).vectorize(x, 8);
    }
    else
    {
        Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
        xSobel.compute_root().gpu_tile(x, y, 16, 16);
        temp.compute_at(xSobel, Var::gpu_blocks());
    }
    return xSobel;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int width, int height, int xmin, int xmax, int ymin, int ymax,
    int x_tile_size = 32, int y_tile_size = 32, bool useGPU = false, float uniquenessRatio = 0.15, int disp12MaxDiff = 1) {

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
    RDom rd(minDisparity, numDisparities);
    Func disp_left("disp_left");
    Func disp("disp");
    if (useGPU) {
        vsum(d, xi, yi, xo, yo) =  sum(diff_T(d, xi, yi+rk, xo, yo));
        cSAD(d, xi, yi, xo, yo) = sum(vsum(d, xi+rk, yi, xo, yo));

        // vsum(d, x, y) = sum(diff(d, x, y+rk));
        // cSAD(d, x, y) = sum(vsum(d, x+rk, y));
        //
        // disp_left(x, y) = {cast<ushort>(minDisparity), cast<ushort>((2<<16)-1)};
        // disp_left(x, y) = tuple_select(
        //         cSAD(rd, x, y) < disp_left(x, y)[1],
        //         {cast<ushort>(rd), cSAD(rd, x, y)},
        //         disp_left(x, y));
        //
        // disp(x, y) = select(
        //     x>xmax-xmin || y>ymax-ymin,
        //     cast<ushort>(FILTERED),
        //     cast<ushort>(disp_left(x, y)[0]));
    }
    else{
        vsum(d, xi, yi, xo, yo) = undef<ushort>();
        vsum(d, xi, 0, xo, yo) = sum(diff_T(d, xi, rk, xo, yo));
        vsum(d, xi, ryi, xo, yo) = vsum(d, xi, ryi-1, xo, yo) + diff_T(d, xi, ryi+win2, xo, yo) - diff_T(d, xi, ryi-win2-1, xo, yo);
        // vsum(d, xi, yi, xo, yo) = diff_T(d, xi, yi, xo, yo);
        // cSAD(d, xi, yi, xo, yo) = vsum(d, xi, yi, xo, yo);
        cSAD(d, xi, yi, xo, yo) = sum(vsum(d, xi+rk, yi, xo, yo));
    }

    disp_left(xi, yi, xo, yo) = {cast<ushort>(minDisparity), cast<ushort>((2<<16)-1)};
    disp_left(xi, yi, xo, yo) = tuple_select(
        cSAD(rd, xi, yi, xo, yo) < disp_left(xi, yi, xo, yo)[1],
        {cast<ushort>(rd), cSAD(rd, xi, yi, xo, yo)},
    disp_left(xi, yi, xo, yo));

    disp(x, y) = select(
            x>xmax-xmin || y>ymax-ymin,
            cast<ushort>(FILTERED),
            cast<ushort>(disp_left(x%x_tile_size, y%y_tile_size, x/x_tile_size, y/y_tile_size)[0]));

    int vector_width = 8;

    // Schedule
    if (useGPU)
    {
        Var x_thread("x_thread"), y_thread("y_thread");
        // disp.compute_root().tile(x, y, xo, yo, xi, yi, x_tile_size, y_tile_size)
        //     .tile(xi, yi, x_thread, y_thread, xi, yi, x_tile_size/16, y_tile_size/16)
        //     .gpu_blocks(xo, yo).gpu_threads(x_thread, y_thread);
        disp.compute_root().tile(x, y, xo, yo, xi, yi, x_tile_size, y_tile_size)
            .gpu_blocks(xo, yo).gpu_threads(xi, yi);

        // reorder storage
        disp_left.reorder_storage(xi, yi, xo, yo);
        diff_T   .reorder_storage(xi, yi, xo, yo, d);
        vsum     .reorder_storage(xi, yi, xo, yo, d);
        cSAD     .reorder_storage(xi, yi, xo, yo, d);

        disp_left.compute_at(disp, Var::gpu_blocks()).gpu_threads(xi, yi)
                 .update().reorder(xi, yi, xo, yo, rd).gpu_threads(xi, yi);

        cSAD.compute_at(disp_left, rd).gpu_threads(xi, yi);
        vsum.compute_at(disp_left, rd).gpu_threads(xi, yi);

        // disp_left.compute_at(disp, Var::gpu_blocks()).tile(xi, yi, x_thread, y_thread, xi, yi, x_tile_size/16, y_tile_size/16).gpu_threads(x_thread, y_thread)
        //          .update().tile(xi, yi, x_thread, y_thread, xi, yi, x_tile_size/16, y_tile_size/16).reorder(xi, yi, x_thread, y_thread, xo, yo, rd).gpu_threads(x_thread, y_thread);
        //
        // cSAD.compute_at(disp_left, rd).tile(xi, yi, x_thread, y_thread, xi, yi, x_tile_size/16, y_tile_size/16).gpu_threads(x_thread, y_thread);
        // vsum.compute_at(disp_left, rd).tile(xi, yi, x_thread, y_thread, xi, yi, x_tile_size/16, y_tile_size/16).gpu_threads(x_thread, y_thread);
        Target target = get_host_target();
        target.set_feature(Target::OpenCL);
        disp.compile_jit(target);
        disp.compile_to_lowered_stmt("disp.html", {}, HTML, target);
    }
    else
    {
        disp.compute_root().tile(x, y, xo, yo, xi, yi, x_tile_size, y_tile_size).reorder(xi, yi, xo, yo)
        .vectorize(xi, vector_width).parallel(xo).parallel(yo);

        // reorder storage
        disp_left.reorder_storage(xi, yi, xo, yo);
        diff_T   .reorder_storage(xi, yi, xo, yo, d);
        vsum     .reorder_storage(xi, yi, xo, yo, d);
        cSAD     .reorder_storage(xi, yi, xo, yo, d);

        disp_left.compute_at(disp, xo).reorder(xi, yi, xo, yo)    .vectorize(xi, vector_width)
                 .update()            .reorder(xi, yi, rd, xo, yo).vectorize(xi, vector_width);

        cSAD.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width);
        vsum.compute_at(disp_left, rd).reorder(xi,  yi, xo, yo, d).vectorize(xi, vector_width)
            .update()                 .reorder(xi, xo, yo, d).vectorize(xi, vector_width);
        vsum.update(1)                .reorder(xi, ryi, xo, yo, d).vectorize(xi, vector_width);
        Target target = get_jit_target_from_environment();
        target.set_feature(Target::SSE41);
        disp.compile_jit(target);
        disp.compile_to_lowered_stmt("disp.html", {}, HTML, target);
    }

    return disp;
}

Image<ushort> stereoBM(Image<uint8_t> left_image, Image<uint8_t> right_image, int SADWindowSize, int minDisparity,
              int numDisparities, int xmin, int xmax, int ymin, int ymax, bool useGPU) {
    Var x("x"), y("y");
    Image<float> gray_left(left_image.width(), left_image.height());
    Image<float> gray_right(left_image.width(), left_image.height());
    for (int y = 0; y < left_image.height(); y++)
    {
        for (int x = 0; x < left_image.width(); x++)
        {
            gray_left(x, y) = (0.2989f*left_image(x, y, 0) + 0.5870f*left_image(x, y, 1) + 0.1140f*left_image(x, y, 2));
            gray_right(x, y) = (0.2989f*right_image(x, y, 0) + 0.5870f*right_image(x, y, 1) + 0.1140f*right_image(x, y, 2));
        }
    }
    Func left("left"), right("right");
    left(x, y) = gray_left(x, y);
    right(x, y) = gray_right(x, y);

    int width = left_image.width();
    int height = left_image.height();

    Func filteredLeft = prefilterXSobel(left, width, height, useGPU);
    Func filteredRight = prefilterXSobel(right, width, height, useGPU);

    int x_tile_size, y_tile_size;
    Func disp;
    if (useGPU)
    {
        x_tile_size = 16, y_tile_size = 16;
        disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
            left_image.width(), left_image.height(), xmin, xmax, ymin, ymax, x_tile_size, y_tile_size, useGPU);
    }
    else
    {
        x_tile_size = 64, y_tile_size = 32;
        disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
            left_image.width(), left_image.height(), xmin, xmax, ymin, ymax, x_tile_size, y_tile_size, useGPU);
    }

    int w = (xmax-xmin)/x_tile_size*x_tile_size+x_tile_size;
    int h = (ymax-ymin)/x_tile_size*x_tile_size+x_tile_size;

    Image<ushort> disp_image(w, h);
    disp.realize(disp_image);

    profile(disp, disp_image, 100);
    return disp_image;
}

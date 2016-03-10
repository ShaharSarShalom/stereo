#include "Halide.h"
#include <limits>
#include "halide_image_io.h"
#include "imageLib.h"

using namespace Halide;

int FILTERED = -16;

void apply_default_schedule(Func F) {
    std::map<std::string,Internal::Function> flist = Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string,Internal::Function>::iterator fit;
    for (fit=flist.begin(); fit!=flist.end(); fit++) {
        // Func f(fit->second);
        Internal::Function f = fit->second;
        Func wrapper(f);
        wrapper.compute_root();
        // if (f.schedule().compute_level().is_inline()){
        //     std::cout << "Warning: applying inline schedule to " << f.name() << std::endl;
        // }
    }
    std::cout << std::endl;
}

Func prefilterXSobel(Func image, int w, int h) {
    Var x("x"), y("y"), c("c");
    Func clamped("clamped");
    clamped(x, y, c) = image(clamp(x, 0, w-1), clamp(y, 0, h-1), c);

    Func temp("temp"), xSobel("xSobel");
    temp(x, y, c) = clamped(x+1, y, c) - clamped(x-1, y, c);
    xSobel(x, y, c) = clamp(temp(x, y-1, c) + 2 * temp(x, y, c) + temp(x, y+1, c), -31, 31);

    // Schedule
    // Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    // xSobel.compute_root();
    // xSobel.tile(x, y, xo, yo, xi, yi, 128, 32);
    // temp.compute_at(xSobel, yi).vectorize(x, 16);
    // xSobel.parallel(yo);
    return xSobel;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int xmin, int xmax, int ymin, int ymax, float uniquenessRatio = 0.15, int disp12MaxDiff = 1) {

    Var x("x"), y("y"), c("c"), d("d");
    Func diff("diff");
    diff(d, x, y) = select(x>=xmin && x<=xmax && y>=ymin && y<=ymax,
        cast<int>(abs(left(x, y, 0) - right(x-d, y, 0)) + abs(left(x, y, 1) - right(x-d, y, 1)) + abs(left(x, y, 2) - right(x-d, y, 2))),
        0);

    Func cSAD("cSAD"), vsum("vsum");
    int win2 = SADWindowSize/2;
    RDom r(-win2, SADWindowSize);
    RDom rx(xmin-win2, xmax - xmin + 1);
    RDom ry(ymin-win2, ymax - ymin + 1 + win2);
    // vsum(d, x, y) = sum(diff(x, y+r, d));
    vsum(d, x, y) = 0;
    vsum(d, x, ry) = select(ry <= win2, vsum(d, x, ry-1) + diff(d, x, ry+win2),
                                        vsum(d, x, ry-1) + diff(d, x, ry+win2) - diff(d, x, ry-win2));

    cSAD(d, x, y) = 0;
    cSAD(d, rx, y) = select(rx <= win2, cSAD(d, rx-1, y) + vsum(d, rx+win2, y),
                                        cSAD(d, rx-1, y) + vsum(d, rx+win2, y) - vsum(d, rx-win2, y));

    RDom rd(minDisparity, numDisparities);
    Func disp_left("disp_left");
    disp_left(x, y) = {minDisparity, INT_MAX};
    disp_left(x, y) = tuple_select(
                             cSAD(rd, x, y) < disp_left(x, y)[1],
                             {rd, cSAD(rd, x, y)},
                             disp_left(x, y)
                             );

    // check unique match
    Func unique("unique");
    unique(x, y) = 1;
    unique(x, y) = select(rd != disp_left(x, y)[0] && cSAD(rd, x, y) <= disp_left(x, y)[1] * (1 + uniquenessRatio), 0, 1);

    // validate disparity by comparing left and right
    // calculate disp2
    Func disp_right("disp_right");
    rx = RDom(xmin, xmax - xmin + 1);
    disp_right(x, y) = {minDisparity, INT_MAX};
    Expr x2 = clamp(rx - disp_left(rx, y)[0], xmin, xmax);
    disp_right(x2, y) = tuple_select(
                             disp_right(x2, y)[1] > disp_left(rx, y)[1],
                             disp_left(rx, y),
                             disp_right(x2, y)
                         );

    Func disp("disp");
    x2 = clamp(x - disp_left(x, y)[0], xmin, xmax);
    disp(x, y) = select(
                    unique(x, y) == 0
                    || !(x >= xmin && x <= xmax && y >= ymin && y<= ymax)
                    || (x2 >= xmin && x2 <= xmax && abs(disp_right(x2, y)[0] - disp_left(x, y)[0]) > disp12MaxDiff),
                    FILTERED,
                    disp_left(x,y)[0]
                 );


    // Schedule
    // Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
    // disp.compute_root().split(y, yo, yi, 64).parallel(yo);
    // unique.compute_inline();
    // disp_right.compute_at(disp, yi).vectorize(x, 16);
    // disp_left.compute_at(disp, yi).vectorize(x, 16);
    //
    // cSAD.compute_at(disp, yi).vectorize(d, 16).reorder(d, x, y);
    // cSAD.reorder_storage(d, x, y);
    // vsum.compute_at(disp, yo).vectorize(d, 16).reorder(d, x, y);
    // vsum.reorder_storage(d, x, y);
    // diff.compute_root().vectorize(x, 16).reorder(x, d, y).parallel(y);
    // diff.reorder_storage(d, x, y);
    return disp;
}


Func stereoMatch(Image<int8_t> left_image, Image<int8_t> right_image, int SADWindowSize, int minDisparity, int numDisparities) {
    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = left_image(x, y, c);
    right(x, y, c) = right_image(x, y, c);

    int width = left_image.width();
    int height = left_image.height();

    Func filteredLeft = prefilterXSobel(left, width, height);
    Func filteredRight = prefilterXSobel(right, width, height);

    /* get valid disparity region */
    int SW2 = SADWindowSize/2;
    int minD = minDisparity, maxD = minDisparity + numDisparities - 1;
    int xmin = maxD + SW2;
    int xmax = left_image.width() - minD - SW2;
    int ymin = SW2;
    int ymax = left_image.height() - SW2;
    Func disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
        xmin, xmax, ymin, ymax);


    disp.compute_root();
    apply_default_schedule(disp);

    Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    disp.realize(width, height, t);
    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    // Image<int> filtered = filteredLeft.realize(width, height, 3);
    // Halide::Tools::save_image(filtered, "filteredLeft.png");
    return disp;
}

#include <unistd.h>
#include <sys/time.h>
unsigned long millisecond_timer(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}

float profile(Func myFunc, int w, int h) {
    myFunc.compile_jit();

    unsigned long s = millisecond_timer();
    int N_TIMES = 1;
    for (int i=0; i<N_TIMES; i++) {
        myFunc.realize(w,h);
    }
    float total_time = float(millisecond_timer()-s);

    float mpixels = float(w*h)/1e6;
    std::cout << "runtime " << total_time/N_TIMES << " ms "
        << " throughput " << (mpixels*N_TIMES)/(total_time/1000) << " megapixels/sec" << std::endl;

    return total_time/N_TIMES;
}

CFloatImage convertHalideImageToFloatImage(Image<int> image) {
    CFloatImage img(image.width(), image.height(), 1);
    for (int x = 0; x < image.width(); x++) {
        for (int y = 0; y < image.height(); y++) {
            float* ptr = (float *) img.PixelAddress(x, y, 0);
            *ptr = (float)image(x,y);
            if (*ptr < 0)
                *ptr = INFINITY;
        }
    }
    return img;
}

static void print_help()
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [-o <disparity_image>] \n");
}

int main(int argc, char **argv) {
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* nodisplay_opt = "--no-display";

    if(argc < 3)
    {
        print_help();
        return 0;
    }
    const char* img1_filename = 0;
    const char* img2_filename = 0;
    const char* disparity_filename = 0;

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    float scale = 1.f;

    for( int i = 1; i < argc; i++ )
    {
        if( argv[i][0] != '-' )
        {
            if( !img1_filename )
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if( strncmp(argv[i], algorithm_opt, strlen(algorithm_opt)) == 0 )
        {
            char* _alg = argv[i] + strlen(algorithm_opt);
            alg = strcmp(_alg, "bm") == 0 ? STEREO_BM :
                  strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  strcmp(_alg, "hh") == 0 ? STEREO_HH :
                  strcmp(_alg, "var") == 0 ? STEREO_VAR : -1;
            if( alg < 0 )
            {
                printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], maxdisp_opt, strlen(maxdisp_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(maxdisp_opt), "%d", &numberOfDisparities ) != 1 ||
                numberOfDisparities < 1 )
            {
                printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
                print_help();
                return -1;
            }
        }
        else if( strncmp(argv[i], blocksize_opt, strlen(blocksize_opt)) == 0 )
        {
            if( sscanf( argv[i] + strlen(blocksize_opt), "%d", &SADWindowSize ) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1 )
            {
                printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if( strcmp(argv[i], "-o" ) == 0 )
            disparity_filename = argv[++i];
        else
        {
            printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }

    if( !img1_filename || !img2_filename )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }

    if ( !disparity_filename)
    {
        printf("Command-line parameter error: output disparity image must be specified\n");
        return -1;
    }

    Image<int8_t> img1 = Halide::Tools::load_image(std::string(img1_filename));
    Image<int8_t> img2 = Halide::Tools::load_image(std::string(img2_filename));
    Func disp = stereoMatch(img1, img2, SADWindowSize, 0, numberOfDisparities);
    // profile(disp, img1.width(), img1.height());
    Image<int> disp_image = disp.realize(img1.width(), img1.height());

    int maxDisparity = numberOfDisparities - 1;

    Image<float> scaled_disp(disp_image.width(), disp_image.height());
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = min(1.f, max(0.f, disp_image(x,y) * 1.0f / maxDisparity));
        }
    };

    Halide::Tools::save_image(scaled_disp, "temp.png");
    WriteFilePFM(convertHalideImageToFloatImage(disp_image), disparity_filename, 1.0f/maxDisparity);
}

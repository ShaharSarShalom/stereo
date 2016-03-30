#include <unistd.h>
#include <sys/time.h>
#include "stereo.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "imageLib.h"

using namespace Halide;

unsigned long millisecond_timer(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}

float profile(Func myFunc, int w, int h) {
    myFunc.compile_jit();

    unsigned long s = millisecond_timer();
    int N_TIMES = 5;
    for (int i=0; i<N_TIMES; i++) {
        myFunc.realize(w,h);
    }
    float total_time = float(millisecond_timer()-s);

    float mpixels = float(w*h)/1e6;
    std::cout << "runtime " << total_time/N_TIMES << " ms "
        << " throughput " << (mpixels*N_TIMES)/(total_time/1000) << " megapixels/sec" << std::endl;

    return total_time/N_TIMES;
}

template <class T>
CFloatImage convertHalideImageToFloatImage(Image<T> image) {
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
    Func disp = stereoBM(img1, img2, SADWindowSize, 0, numberOfDisparities);
    profile(disp, img1.width(), img1.height());
    Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    Image<int> disp_image = disp.realize(img1.width(), img1.height(), t);

    int maxDisparity = numberOfDisparities - 1;

    Image<float> scaled_disp(disp_image.width(), disp_image.height());
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = min(1.f, max(0.f, disp_image(x,y) * 1.0f / maxDisparity));
        }
    };

    int filename_len = std::strlen(disparity_filename);
    if (filename_len >= 4 && strcmp(disparity_filename + filename_len - 4, ".png") == 0) {
        Halide::Tools::save_image(scaled_disp, disparity_filename);
    }
    else {
        WriteFilePFM(convertHalideImageToFloatImage<int>(disp_image), disparity_filename, 1.0f/maxDisparity);
    }
}

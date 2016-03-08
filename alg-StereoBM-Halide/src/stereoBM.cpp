#include "Halide.h"
#include <limits>
#include "halide_image_io.h"

using namespace Halide;

int FILTERED = -16;

void apply_default_schedule(Func F) {
    std::map<std::string,Internal::Function> flist = Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string,Internal::Function>::iterator fit;
    for (fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        std::cout << "Warning: applying default schedule to " << f.name() << std::endl;
    }
    std::cout << std::endl;
}

Func prefilterNorm(Func image, int winsize, int w, int h) {
    Var x("x"), y("y"), c("c");
    Func intensity("intensity");
    intensity(x, y) = (image(x, y, 0) + image(x, y, 1) + image(x, y, 2))/3.0f;
    Func clamped("clamped");
    clamped(x, y) = intensity(clamp(x, 0, w-1), clamp(y, 0, h-1));

    Func vsum("vsum"), hsum("sum");
    int win2 = winsize/2;
    RDom rw(-win2, win2);
    vsum(x, y) = 0.0f;
    RDom rx(0, w), ry(0, h);
    vsum(x, ry) = select(ry == 0, sum(clamped(x, rw)), vsum(x, ry-1) - clamped(x, ry-win2-1) + clamped(x, ry + win2));

    hsum(x, y) = 0.0f;
    hsum(rx, y) = select(rx == 0, sum(vsum(rw, y)), hsum(rx-1, y) - vsum(rx-win2-1, y) + vsum(rx + win2, y));

    Func factor("factor");
    Expr x_size = min(w - x + win2, min(x + win2, winsize));
    Expr y_size = min(h - y + win2, min(y + win2, winsize));
    factor(x, y, c) = hsum(x,y) / (x_size * y_size)/ hsum(x, y);
    Func filtered("filtered");
    filtered(x, y, c) = image(x, y, c) / factor(x, y, c);
    return filtered;
}

Func findStereoCorrespondence(Func left, Func right, int SADWindowSize, int minDisparity, int numDisparities,
    int xmin, int xmax, int ymin, int ymax, float uniquenessRatio = 0.15) {
    Func cost("cost"), diff("diff");
    Var x("x"), y("y"), c("c"), d("d");

    diff(x, y, d) = abs(left(x, y, 0) - right(x+d, y, 0)) + abs(left(x, y, 1) - right(x+d, y, 1)) + abs(left(x, y, 2) - right(x+d, y, 2));
    int w2 = SADWindowSize / 2;
    RDom r(-w2, w2);
    Func vsum("vsum");
    RDom rx(xmin, xmax-xmin+1), ry(ymin, ymax-ymin+1);
    vsum(x, y, d) = 0.0f;
    vsum(x, ry, d) = select(ry == ymin, sum(diff(x, r, d)), vsum(x, ry-1, d) - diff(x, ry-w2-1, d) + diff(x, ry+w2, d));
    cost(x, y, d) = 0.0f;
    cost(rx, y, d) = select(rx == xmin, sum(vsum(r, y, d)), cost(rx-1, y, d) - vsum(rx-w2-1, y, d) + vsum(rx+w2, y, d));

    RDom rd(minDisparity, numDisparities);
    Func minsad("minsad");
    //  second_min_sad, second_min_cost, minsad, mincost,
    minsad(x, y) = Tuple(FILTERED, std::numeric_limits<float>::infinity(), minDisparity, cost(x, y, minDisparity));
    Expr compareMin = cost(x,y,rd) < minsad(x,y)[3];
    Expr compareSecondMin = cost(x,y,rd) < minsad(x,y)[2];
    minsad(x, y) = tuple_select({compareMin, compareMin, compareMin, compareMin},
                                {minsad(x,y)[0], minsad(x,y)[1], rd, cost(x,y,rd)},
                                {select(compareSecondMin, rd, minsad(x,y)[2]),
                                 select(compareSecondMin, cost(x,y,rd), minsad(x,y)[3]),
                                 minsad(x,y)[2], minsad(x,y)[3]});
    Func disp("disparity");
    disp(x, y) = FILTERED;
    RDom ri(xmin, xmax-xmin+1, ymin, ymax-ymin+1);
    disp(ri.x, ri.y) = select(minsad(ri.x, ri.y)[1] < minsad(ri.x,ri.y)[3] * (1+uniquenessRatio), FILTERED, minsad(ri.x, ri.y)[0]);

    return disp;
}

// check whether machine is little endian
int littleendian()
{
    int intval = 1;
    unsigned char *uval = (unsigned char *)&intval;
    return uval[0] == 1;
}


Func stereoMatch(Image<float> left_image, Image<float> right_image, int SADWindowSize, int minDisparity, int numDisparities) {
    Var x("x"), y("y"), c("c");
    Func left("left"), right("right");
    left(x, y, c) = left_image(x, y, c);
    right(x, y, c) = right_image(x, y, c);

    Func filteredLeft = prefilterNorm(left, 9, left_image.width(), left_image.height());
    Func filteredRight = prefilterNorm(left, 9, right_image.width(), right_image.height());

    /* get valid disparity region */
    int SW2 = SADWindowSize/2;
    int minD = minDisparity, maxD = minDisparity + numDisparities - 1;
    int xmin = maxD + SW2;
    int xmax = left_image.width() - minD - SW2;
    int ymin = SW2;
    int ymax = left_image.height() - SW2;
    Func disp = findStereoCorrespondence(filteredLeft, filteredRight, SADWindowSize, minDisparity, numDisparities,
        xmin, xmax, ymin, ymax);

    apply_default_schedule(disp);
    return disp;
}

// 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
void WriteFilePFM(Image<int> image, const char* filename, float scalefactor=1/255.0)
{
    // Open the file
    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
	exit(1);
    }

    // sign of scalefact indicates endianness, see pfms specs
    if (littleendian())
	scalefactor = -scalefactor;

    int width = image.width(), height = image.height();

    // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
    fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

    int n = width;
    // write rows -- pfm stores rows in inverse order!
    for (int y = height-1; y >= 0; y--) {
    	float ptr[width];
    	// change invalid pixels (which seem to be represented as -10) to INF
    	for (int x = 0; x < width; x++) {
    		ptr[x] = image(x,y) < 0 ? INFINITY : image(x,y);
    	}
    	if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
    	    fprintf(stderr, "WriteFilePFM: problem writing data\n");
    	    exit(1);
    	}
    }

    // close file
    fclose(stream);
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
    int N_TIMES = 10;
    for (int i=0; i<N_TIMES; i++) {
        myFunc.realize(w,h);
    }
    float total_time = float(millisecond_timer()-s);

    float mpixels = float(w*h)/1e6;
    std::cout << "runtime " << total_time/N_TIMES << " ms "
        << " throughput " << (mpixels*N_TIMES)/(total_time/1000) << " megapixels/sec" << std::endl;

    return total_time/N_TIMES;
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
                numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
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

    Image<float> img1 = Halide::Tools::load_image(std::string(img1_filename));
    Image<float> img2 = Halide::Tools::load_image(std::string(img2_filename));
    Func disp = stereoMatch(img1, img2, SADWindowSize, 0, numberOfDisparities);
    profile(disp, img1.width(), img1.height());
    Image<int> disp_image = disp.realize(img1.width(), img1.height());

    int maxDisparity = numberOfDisparities - 1;

    Halide::Tools::save_image(disp_image, std::string(disparity_filename));
    WriteFilePFM(disp_image, disparity_filename, 1.0f/maxDisparity);
}

#include <unistd.h>
#include <sys/time.h>
#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;

unsigned long millisecond_timer(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long)(t.tv_usec/1000 + t.tv_sec*1000);
}

float profile(Func myFunc, int w, int h, int n_times) {
    myFunc.compile_jit();

    unsigned long s = millisecond_timer();
    for (int i=0; i<n_times; i++) {
        myFunc.realize(w,h);
    }
    float total_time = float(millisecond_timer()-s);

    float mpixels = float(w*h)/1e6;
    std::cout << "runtime " << total_time/n_times << " ms "
        << " throughput " << (mpixels*n_times)/(total_time/1000) << " megapixels/sec" << std::endl;

    return total_time/n_times;
}

float profile(Func myFunc, int w, int h, int c, int n_times) {
    myFunc.compile_jit();

    unsigned long s = millisecond_timer();
    for (int i=0; i<n_times; i++) {
        myFunc.realize(w,h,c);
    }
    float total_time = float(millisecond_timer()-s);

    float mpixels = float(w*h*c)/1e6;
    std::cout << "runtime " << total_time/n_times << " ms "
        << " throughput " << (mpixels*n_times)/(total_time/1000) << " megapixels/sec" << std::endl;

    return total_time/n_times;
}

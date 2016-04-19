#include "Halide.h"
#include "halide_image_io.h"
#include "stereo.h"
#include <limits>

void apply_default_schedule(Func F) {
    std::map<std::string,Internal::Function> flist = Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string,Internal::Function>::iterator fit;
    for (fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        // std::cout << "Warning: applying default schedule to " << f.name() << std::endl;
    }
    std::cout << std::endl;
}

Func mean(Func input, int r) {
    float scale = 1.0f/(2*r+1)/(2*r+1);
    RDom k(-r, 2*r+1);

    Var x("x"), y("y"), d("d");
    Func hsum, m;
    if (input.dimensions() == 2)
    {
        hsum(x, y) = sum(input(x + k, y));
        m(x, y) = sum(hsum(x, y+k)) * scale;
    }
    else{
        hsum(x, y, d) = sum(input(x + k, y, d));
        m(x, y, d) = sum(hsum(x, y+k, d)) * scale;
    }
    return m;
}

Func guidedFilter_gray(Func I, Func p, int r, float epsilon) {

    Var x("x"), y("y"), d("d");
    float scale = 1.0f/(2*r+1)/(2*r+1);

    Func mu = mean(I, r);
    Func square("square");
    square(x, y) = I(x, y) * I(x, y);
    Func s_m = mean(square, r);

    Func sigma("sigma");
    sigma(x, y) = s_m(x, y) - pow(mu(x, y), 2);

    Func prod("prod");
    prod(x, y, d) = I(x, y) * p(x, y, d);
    Func prod_m = mean(prod, r);

    Func p_m = mean(p, r);

    Func a("a"), b("b");
    a(x, y, d) = (prod_m(x, y, d) - mu(x, y) * p_m(x, y, d))/(sigma(x, y) + epsilon);
    b(x, y, d) = p_m(x, y, d) - a(x, y, d) * mu(x, y);

    Func a_m = mean(a, r);
    Func b_m = mean(b, r);

    Func q("q");
    q(x, y, d) = a_m(x, y, d) * I(x, y) + b_m(x, y, d);
    apply_default_schedule(q);
    return q;
}

Func guidedFilter(Func I, Func p, int r, float epsilon) {

    Var x("x"), y("y"), d("d");
    float scale = 1.0f/(2*r+1)/(2*r+1);

    Func mu = mean(I, r);
    Func square("square");
    square(x, y) = I(x, y) * I(x, y);
    Func s_m = mean(square, r);

    Func sigma("sigma");
    sigma(x, y) = s_m(x, y) - pow(mu(x, y), 2);

    Func prod("prod");
    prod(x, y, d) = I(x, y) * p(x, y, d);
    Func prod_m = mean(prod, r);

    Func p_m = mean(p, r);

    Func a("a"), b("b");
    a(x, y, d) = (prod_m(x, y, d) - mu(x, y) * p_m(x, y, d))/(sigma(x, y) + epsilon);
    b(x, y, d) = p_m(x, y, d) - a(x, y, d) * mu(x, y);

    Func a_m = mean(a, r);
    Func b_m = mean(b, r);

    Func q("q");
    q(x, y, d) = a_m(x, y, d) * I(x, y) + b_m(x, y, d);
    apply_default_schedule(q);
    return q;

}

Func gradientX(Func image) {
    Var x("x"), y("y");
    Func temp("temp"), gradient_x("gradient_x");
    temp(x, y) = image(x+1, y) - image(x-1, y);
    gradient_x(x, y) = temp(x, y-1) + 2 * temp(x, y) + temp(x, y+1);
    return gradient_x;
}

Func stereoGF(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad){
    Var x("x"), y("y"), d("d");
    Func left_gradient = gradientX(left);
    Func right_gradient = gradientX(right);

    Func cost_left("cost_left"), cost_right("cost_right");
    cost_left(x, y, d) = (1 - alpha) * clamp(abs(left(x, y) - right(x-d, y)), 0, threshColor) +
                          alpha * clamp(abs(left_gradient(x, y) - right_gradient(x-d, y)), 0, threshGrad);
    cost_right(x, y, d) = (1 - alpha) * clamp(abs(right(x, y) - left(x+d, y)), 0, threshColor) +
                          alpha * clamp(abs(right_gradient(x, y) - left_gradient(x+d, y)), 0, threshGrad);

    Func filtered_left = guidedFilter_gray(left, cost_left, r, epsilon);
    Func filtered_right = guidedFilter_gray(right, cost_right, r, epsilon);

    RDom rd(0, numDisparities);
    Func disp_left("disp_left"), disp_right("disp_right");
    disp_left(x, y) = {0, INFINITY};
    disp_left(x, y) = tuple_select(
            filtered_left(x, y, rd) < disp_left(x, y)[1],
            {rd, filtered_left(x, y, rd)},
            disp_left(x, y));

    disp_right(x, y) = {0, INFINITY};
    disp_right(x, y) = tuple_select(
            filtered_right(x, y, rd) < disp_right(x, y)[1],
            {rd, filtered_right(x, y, rd)},
            disp_right(x, y));

    Func disp("disp");
    Expr disp_val = disp_left(x, y)[0];
    disp(x, y) = select(x > disp_val && abs(disp_right(clamp(x - disp_val, 0, width-1), y)[0] - disp_val) < 1, disp_val, FILTERED);
    apply_default_schedule(disp);
    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    return disp;
}

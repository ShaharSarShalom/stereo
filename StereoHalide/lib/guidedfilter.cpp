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

    Var x("x"), y("y"), c("c"), d("d");
    Func hsum, m;
    if (input.dimensions() == 2)
    {
        hsum(x, y) = sum(input(x + k, y));
        m(x, y) = sum(hsum(x, y+k)) * scale;
    }
    else if (input.dimensions() == 3){
        hsum(x, y, d) = sum(input(x + k, y, d));
        m(x, y, d) = sum(hsum(x, y+k, d)) * scale;
    }
    else{
        hsum(x, y, c, d) = sum(input(x + k, y, c, d));
        m(x, y, c, d) = sum(hsum(x, y+k, c, d)) * scale;
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

    Var x("x"), y("y"), c("c"), d("d");
    float scale = 1.0f/(2*r+1)/(2*r+1);

    Func mu = mean(I, r);
    Func square("square");
    Func ind2pair("ind2pair"), pair2ind("pair2ind");
    ind2pair(c) = {0,c};
    ind2pair(3) = {1,1};
    ind2pair(4) = {1,2};
    ind2pair(5) = {2,2};
    pair2ind(c, d) = undef<int>();
    pair2ind(0, 0) = 0;
    pair2ind(0, 1) = 1;
    pair2ind(0, 2) = 2;
    pair2ind(1, 0) = 1;
    pair2ind(1, 1) = 3;
    pair2ind(1, 2) = 4;
    pair2ind(2, 0) = 2;
    pair2ind(2, 1) = 4;
    pair2ind(2, 2) = 5;


    Expr row = clamp(ind2pair(c)[0], 0, 2), col = clamp(ind2pair(c)[1], 0, 2);
    square(x, y, c) = I(x, y, row) * I(x, y, col);
    Func s_m = mean(square, r);

    Func sigma("sigma");
    sigma(x, y, c) = s_m(x, y, c) - mu(x, y, row) * mu(x, y, col);
    Expr a11 = sigma(x, y, 0) + epsilon, a12 = sigma(x, y, 1), a13 = sigma(x, y, 2);
    Expr a22 = sigma(x, y, 3) + epsilon, a23 = sigma(x, y, 4);
    Expr a33 = sigma(x, y, 5) + epsilon;

    Func inv_("inv_"), inv("inv");
    inv_(x, y, c) = undef<float>();
    inv_(x, y, 0) = a22 * a33 - a23 * a23;
    inv_(x, y, 1) = a13 * a23 - a12 * a33;
    inv_(x, y, 2) = a12 * a23 - a22 * a13;
    inv_(x, y, 3) = a11 * a33 - a13 * a13;
    inv_(x, y, 4) = a13 * a12 - a11 * a23;
    inv_(x, y, 5) = a11 * a22 - a12 * a12;
    Expr det = a11 * inv_(x, y, 0) + a12 * inv_(x, y, 1) + a13 * inv_(x, y, 2);
    inv(x, y, c) = inv_(x, y, c) / det;
    Expr check1 = a11 * inv(x, y, 0) + a12 * inv(x, y, 1) + a13 * inv(x, y, 2);
    Expr check2 = a12 * inv(x, y, 1) + a22 * inv(x, y, 3) + a23 * inv(x, y, 4);
    Expr check3 = a13 * inv(x, y, 2) + a23 * inv(x, y, 4) + a33 * inv(x, y, 5);

    Func prod("prod");
    prod(x, y, c, d) = I(x, y, c) * p(x, y, d);
    Func prod_m = mean(prod, r);
    Func p_m = mean(p, r);

    Func temp("temp");
    temp(x, y, c, d) = prod_m(x, y, c, d) - mu(x, y, c) * p_m(x, y, d);

    Func a("a"), b("b");
    RDom k(0, 3, "k");
    a(x, y, c, d) = sum(inv(x, y, clamp(pair2ind(c, k), 0, 5)) * temp(x, y, k, d));
    b(x, y, d) = p_m(x, y, d) - sum(a(x, y, k, d) * mu(x, y, k));

    Func a_m = mean(a, r);
    Func b_m = mean(b, r);

    Func q("q");
    q(x, y, d) = sum(a_m(x, y, k, d) * I(x, y, k)) + b_m(x, y, d);
    apply_default_schedule(q);
    return q;

}

Func gradientX(Func image) {
    Var x("x"), y("y"), c("c");
    Func temp("temp"), gradient_x("gradient_x"), gray("gray");
    if (image.dimensions() == 2)
    {
        gray(x, y) = image(x, y);
    }
    else
    {
        gray(x, y) = 0.2989f*image(x, y, 0) + 0.5870f*image(x, y, 1) + 0.1140f*image(x, y, 2);
    }
    // temp(x, y) = 0.5f * (gray(x+1, y) - gray(x-1, y));
    // gradient_x(x, y) = 0.25f * (temp(x, y-1) + 2 * temp(x, y) + temp(x, y+1));
    gradient_x(x, y) = 0.5f*(gray(x+1, y) - gray(x-1, y));
    return gradient_x;
}

Func stereoGF(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad){
    Var x("x"), y("y"), c("c"), d("d");
    Func left_gradient = gradientX(left);
    Func right_gradient = gradientX(right);

    Func cost_left("cost_left"), cost_right("cost_right");
    Func diff("diff");
    diff(x, y, c, d) = abs(left(x, y, c) - right(x-d, y, c));
    cost_left(x, y, d) = (1 - alpha) * clamp((diff(x, y, 0, d) + diff(x, y, 1, d) + diff(x, y, 2, d))/3, 0, threshColor) +
                          alpha * clamp(abs(left_gradient(x, y) - right_gradient(x-d, y)), 0, threshGrad);
    cost_right(x, y, d) = cost_left(x + d, y, d);

    Func filtered_left = guidedFilter(left, cost_left, r, epsilon);
    Func filtered_right = guidedFilter(right, cost_right, r, epsilon);

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
    disp(x, y) = select(x > disp_val && abs(disp_right(clamp(x - disp_val, 0, width-1), y)[0] - disp_val) < 1, disp_val, -1);
    // disp(x, y) = disp_left(x, y)[0];
    apply_default_schedule(disp);
    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    return disp;
}


/**************************************************************************************/
/******************** functions with schedule *****************************************/


/******************************* schedule mean *****************************/
/* return I_copy if copies I
* this is to enable schedule functions at I_copy
*/
Func meanWithSchedule(Func I, Func meanI, int r, Var compute_level, bool c_innermost, bool copy = false)
{
    float scale = 1.0f/(2*r+1)/(2*r+1);
    Var x("x"), y("y"), c(I.function().args()[2]), d("d");
    Func I_copy("copy"), I_vsum("vsum");
    RDom k(-r, 2*r+1);
    int vector_width = 8;

    if (I.dimensions() == 3)
    {
        I_copy(x, y, c) = I(x, y, c);
        I_vsum(x, y, c) = sum(I_copy(x+k, y, c));
        meanI(x, y, c) = sum(I_vsum(x, y+k, c)) * scale;

        /*************** Schedule *******************/
        I_vsum.compute_at(meanI, compute_level).unroll(c).vectorize(x, vector_width);
        if  (copy)
        {
            I_copy.compute_at(I_vsum, y).unroll(c).vectorize(x, vector_width);
        }
        else
        {
            I.compute_at(I_vsum, y).unroll(c).vectorize(x, vector_width);
        }
        if (c_innermost)
        {
            I_vsum.reorder(c, x, y);
            I_copy.reorder(c, x, y);
            I.reorder(c, x, y);
        }

    }
    else if (I.dimensions() == 4)
    {
        I_copy(x, y, c, d) = I(x, y, c, d);
        I_vsum(x, y, c, d) = sum(I_copy(x+k, y, c, d));
        meanI(x, y, c, d) = sum(I_vsum(x, y+k, c, d)) * scale;

        /*************** Schedule *******************/
        I_vsum.compute_at(meanI, compute_level).unroll(c).unroll(d).vectorize(x, vector_width);
        if  (copy)
        {
            I_copy.compute_at(I_vsum, y).unroll(c).unroll(d).vectorize(x, vector_width);
        }
        else
        {
            I.compute_at(I_vsum, y).unroll(c).unroll(d).vectorize(x, vector_width);
        }
        if (c_innermost)
        {
            I_vsum.reorder(c, x, y, d);
            I_copy.reorder(c, x, y, d);
            I.reorder(c, x, y, d);
        }
    }

    if (copy)
    {
        return I_copy;
    }
    else
    {
        return I;
    }
}

void guidanceImageProcessing(Func I, Func& meanI, Func& inv, int r, float eps)
{
    Var x("x"), y("y"), c("c"), d("d");
    float scale = 1.0f/(2*r+1)/(2*r+1);

    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    meanWithSchedule(I, meanI, r, Var::outermost(), false/*c_innermost*/, true/*copy*/);

    Func square("square");
    Func ind2pair("ind2pair");
    ind2pair(c) = {0,c};
    ind2pair(3) = {1,1};
    ind2pair(4) = {1,2};
    ind2pair(5) = {2,2};

    Expr row = clamp(ind2pair(c)[0], 0, 2), col = clamp(ind2pair(c)[1], 0, 2);
    square(x, y, c) = I(x, y, row) * I(x, y, col);
    Func s_m("s_m");
    meanWithSchedule(square, s_m, r, Var::outermost(), true/*c_inndermost*/);

    Func sigma("sigma");
    sigma(x, y, c) = s_m(x, y, c) - meanI(x, y, row) * meanI(x, y, col);
    sigma(x, y, 0) = sigma(x, y, 0) + eps;
    sigma(x, y, 3) = sigma(x, y, 3) + eps;
    sigma(x, y, 5) = sigma(x, y, 5) + eps;
    Expr a11 = sigma(x, y, 0), a12 = sigma(x, y, 1), a13 = sigma(x, y, 2);
    Expr a22 = sigma(x, y, 3), a23 = sigma(x, y, 4);
    Expr a33 = sigma(x, y, 5);

    Func inv_("inv_");
    inv_(x, y, c) = undef<float>();
    inv_(x, y, 0) = a22 * a33 - a23 * a23;
    inv_(x, y, 1) = a13 * a23 - a12 * a33;
    inv_(x, y, 2) = a12 * a23 - a22 * a13;
    inv_(x, y, 3) = a11 * a33 - a13 * a13;
    inv_(x, y, 4) = a13 * a12 - a11 * a23;
    inv_(x, y, 5) = a11 * a22 - a12 * a12;
    inv_(x, y, 6) = a11 * inv_(x, y, 0) + a12 * inv_(x, y, 1) + a13 * inv_(x, y, 2);
    inv(x, y, c) = inv_(x, y, c) / inv_(x, y, 6);

    /****************** Schedule ********************/
    int vector_width = 8;

    // inv.compute_root().tile(x, y, xo, yo, xi, yi, 32, 32).reorder(c, xi, yi, xo, yo).vectorize(xi, vector_width);
    inv_.compute_at(inv, x).reorder(c, x, y).unroll(c).vectorize(x, vector_width);
    inv_.update().vectorize(x, vector_width);
    inv_.update(1).vectorize(x, vector_width);
    inv_.update(2).vectorize(x, vector_width);
    inv_.update(3).vectorize(x, vector_width);
    inv_.update(4).vectorize(x, vector_width);
    inv_.update(5).vectorize(x, vector_width);
    inv_.update(6).vectorize(x, vector_width);
    sigma.compute_at(inv, x).reorder(c, x, y).unroll(c).vectorize(x, vector_width);
    s_m.compute_at(inv, Var::outermost()).reorder(c, x, y).unroll(c).vectorize(x, vector_width);

    // meanI.compute_root().tile(x, y, xo, yo, xi, yi, 32, 32).reorder(xi, yi, xo, yo, c).vectorize(xi, vector_width);
    ind2pair.compute_root();
}

void filteringCost(Func I, Func p, Func meanI, Func meanP, Func inv, Func& filtered, int r)
{
    Var x("x"), y("y"), c("c"), d("d");
    Func pair2ind("pair2ind");
    pair2ind(c, d) = undef<int>();
    pair2ind(0, 0) = 0;
    pair2ind(0, 1) = 1;
    pair2ind(0, 2) = 2;
    pair2ind(1, 0) = 1;
    pair2ind(1, 1) = 3;
    pair2ind(1, 2) = 4;
    pair2ind(2, 0) = 2;
    pair2ind(2, 1) = 4;
    pair2ind(2, 2) = 5;

    Func prod("prod");
    prod(x, y, c, d) = I(x, y, c) * p(x, y, d);
    Func prod_m("prod_m");
    meanWithSchedule(prod, prod_m, r, d, true);

    Func a_factor("a_factor");
    a_factor(x, y, c, d) = prod_m(x, y, c, d) - meanI(x, y, c) * meanP(x, y, d);

    Func ab("ab");
    RDom k(0, 3, "k");
    ab(x, y, c, d) = undef<float>();
    ab(x, y, 0, d) = sum(inv(x, y, clamp(pair2ind(0, k), 0, 5)) * a_factor(x, y, k, d));
    ab(x, y, 1, d) = sum(inv(x, y, clamp(pair2ind(1, k), 0, 5)) * a_factor(x, y, k, d));
    ab(x, y, 2, d) = sum(inv(x, y, clamp(pair2ind(2, k), 0, 5)) * a_factor(x, y, k, d));
    ab(x, y, 3, d) = meanP(x, y, d) - ab(x, y, 0, d) * meanI(x, y, 0) - ab(x, y, 1, d) * meanI(x, y, 1) - ab(x, y, 2, d) * meanI(x, y, 2);

    Func ab_m("ab_m");
    Func ab_copy = meanWithSchedule(ab, ab_m, r, d, true, true /*copy*/);

    filtered(x, y, d) = ab_m(x, y, 3, d)
                      + ab_m(x, y, 0, d) * I(x, y, 0)
                      + ab_m(x, y, 1, d) * I(x, y, 1)
                      + ab_m(x, y, 2, d) * I(x, y, 2);

    /************************** Schedule *************************/
    int vector_width = 8;
    ab_m.compute_at(filtered, d).reorder(x, y, c, d).unroll(c).unroll(d).vectorize(x, vector_width);

    ab.compute_at(ab_copy, x);
    ab.update(0).reorder(x, y, d).unroll(d).vectorize(x, vector_width);
    ab.update(1).reorder(x, y, d).unroll(d).vectorize(x, vector_width);
    ab.update(2).reorder(x, y, d).unroll(d).vectorize(x, vector_width);
    ab.update(3).reorder(x, y, d).unroll(d).vectorize(x, vector_width);

    a_factor.compute_at(ab_copy, x).reorder(c, x, y, d).unroll(c).unroll(d).vectorize(x, vector_width);

    prod_m.compute_at(filtered, d).reorder(c, x, y, d).unroll(c).unroll(d).vectorize(x, vector_width);
    pair2ind.compute_root();
}

Func stereoGF_scheduled(Func left, Func right, int width, int height, int r, float epsilon, int numDisparities, float alpha, float threshColor, float threshGrad)
{
    Var x("x"), y("y"), c("c"), d("d");
    Func left_gradient = gradientX(left);
    Func right_gradient = gradientX(right);

    Func cost("cost");
    RDom rc(0, 3, "rc");
    cost(x, y, d) = (1 - alpha) * clamp(sum(abs(left(x, y, rc) - right(x-d, y, rc)))/3, 0, threshColor) +
                          alpha * clamp(abs(left_gradient(x, y) - right_gradient(x-d, y)), 0, threshGrad);

    Func mean_left("mean_left"), mean_right("mean_right"), inv_left("inv_left"), inv_right("inv_right");
    guidanceImageProcessing(left, mean_left, inv_left, r, epsilon);
    guidanceImageProcessing(right, mean_right, inv_right, r, epsilon);

    Func mean_cost("mean_cost");
    meanWithSchedule(cost, mean_cost, r, d, false/*d innermost*/, true/*copy*/);

    Func filtered_left("filtered_left");
    filteringCost(left, cost, mean_left, mean_cost, inv_left, filtered_left, r);

    RDom rd(0, numDisparities, "rd");
    Func disp_left("disp_left"), disp_right("disp_right");
    disp_left(x, y) = {0, INFINITY};
    disp_left(x, y) = tuple_select(
            filtered_left(x, y, rd) < disp_left(x, y)[1],
            {rd, filtered_left(x, y, rd)},
            disp_left(x, y));
    //
    // disp_right(x, y) = {0, INFINITY};
    // disp_right(x, y) = tuple_select(
    //         filtered_right(x, y, rd) < disp_right(x, y)[1],
    //         {rd, filtered_right(x, y, rd)},
    //         disp_right(x, y));
    //
    Func disp("disp");
    Expr disp_val = disp_left(x, y)[0];
    // disp(x, y) = select(x > disp_val && abs(disp_right(clamp(x - disp_val, 0, width-1), y)[0] - disp_val) < 1, disp_val, -1);
    disp(x, y) = disp_left(x, y)[0];

    /****************************** Schedule ****************************/
    int vector_width = 8;
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    disp.compute_root().vectorize(x, vector_width);
    disp_left.compute_root().vectorize(x, vector_width)
             .update().tile(x, y, xo, yo, xi, yi, 32, 32).reorder(xi, yi, xo, yo, rd).vectorize(xi, vector_width);

    // filtered_left.compute_at(disp_left, rd).reorder(x, y, d).vectorize(x, vector_width);
    filtered_left.compute_at(disp_left, rd).tile(x, y, xo, yo, xi, yi, 128, 64)
                 .reorder(xi, yi, d, xo, yo).vectorize(xi, vector_width);
    mean_cost.compute_at(filtered_left, d).vectorize(x, vector_width);
    cost.compute_at(filtered_left, d).vectorize(x, vector_width);

    mean_left.compute_root().vectorize(x, vector_width);
    inv_left.compute_root().reorder(c, x, y).vectorize(x, vector_width);

    left.compute_root().vectorize(x, vector_width);
    right.compute_root().vectorize(x, vector_width);
    left_gradient.compute_root().vectorize(x, vector_width);
    right_gradient.compute_root().vectorize(x, vector_width);
    return disp;
}

void guidedFilterTest()
{
    Image<float> im0 = Halide::Tools::load_image("../../trainingQ/Teddy/im0.png");
    Var x("x"), y("y"), c("c");
    // Func I("I"), meanI("meanI"), inv("inv");
    // I(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    // guidanceImageProcessing(I, meanI, inv, 9, 0.01);
    // Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    // inv.compile_to_lowered_stmt("inv.html", {}, HTML, t);
    // inv.realize(im0.width(), im0.height(), 6, t);
    // profile(inv, im0.width(), im0.height(), 6, 50);

    Image<float> im1 = Halide::Tools::load_image("../../trainingQ/Teddy/im1.png");
    Func left("left"), right("right");
    left(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    right(x, y, c) = im1(clamp(x, 0, im1.width()-1), clamp(y, 0, im1.height()-1), c);
    Func disp = stereoGF_scheduled(left, right, im0.width(), im0.height(), 9, 0.01, 60, 0.9, 0.0028, 0.008);

    Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    Image<int> disp_image = disp.realize(im0.width(), im0.height(), t);

    Image<float> scaled_disp(im0.width(), im0.height());
    for (int y = 0; y < disp_image.height(); y++) {
        for (int x = 0; x < disp_image.width(); x++) {
            scaled_disp(x, y) = std::min(1.f, std::max(0.f, disp_image(x,y) * 1.0f / 59));
        }
    };

    Halide::Tools::save_image(scaled_disp, "disp.png");
    profile(disp, im0.width(), im0.height(), 10);
}

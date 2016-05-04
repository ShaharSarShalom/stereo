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

void check_inline_schedule(Func F) {
    std::map<std::string,Internal::Function> flist = Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string,Internal::Function>::iterator fit;
    for (fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        if (fit->second.schedule().compute_level().is_inline())
        {
            std::cout << "Warning: applying inline schedule to " << f.name() << std::endl;
        }
    }
    std::cout << std::endl;
}

Func mean(Func input, int r, bool c_innermost = false, bool schedule_input = true) {
    float scale = 1.0f/(2*r+1)/(2*r+1);
    RDom k(-r, 2*r+1);

    Var x("x"), y("y"), c("c"), d("d");
    Func hsum("hsum"), m("mean");
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

    /***** schedule ****/
    hsum.compute_at(m, Var::outermost());
    if (schedule_input)
    {
        input.compute_at(hsum, y);
    }
    if (c_innermost)
    {
        if (input.dimensions() == 4)
        {
            m.reorder(c, x, y, d);
        }
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

    Func mu = mean(I, r, false, false);
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
    Func p_m = mean(p, r, false, false);

    Func a_factor("a_factor");
    a_factor(x, y, c, d) = prod_m(x, y, c, d) - mu(x, y, c) * p_m(x, y, d);

    Func ab("ab");
    RDom k(0, 3, "k");
    RDom rc(0, 4, "rc");
    ab(x, y, c, d) = undef<float>();
    ab(x, y, rc, d) = select(rc == 3,
                            p_m(x, y, d) - ab(x, y, 0, d) * mu(x, y, 0)
                                           - ab(x, y, 1, d) * mu(x, y, 1)
                                           - ab(x, y, 2, d) * mu(x, y, 2),
                            inv(x, y, clamp(pair2ind(rc, 0), 0, 5)) * a_factor(x, y, 0, d)
                          + inv(x, y, clamp(pair2ind(rc, 1), 0, 5)) * a_factor(x, y, 1, d)
                          + inv(x, y, clamp(pair2ind(rc, 2), 0, 5)) * a_factor(x, y, 2, d) );

    Func ab_m = mean(ab, r, true/*c_innermost*/);

    Func q("q");
    q(x, y, d) = ab_m(x, y, 3, d)
                  + ab_m(x, y, 0, d) * I(x, y, 0)
                  + ab_m(x, y, 1, d) * I(x, y, 1)
                  + ab_m(x, y, 2, d) * I(x, y, 2);

    ind2pair.compute_root();
    pair2ind.compute_root();
    ab_m.compute_at(q, Var::outermost());
    ab.update().unroll(rc).reorder(rc, x, y, d);
    a_factor.compute_at(ab, x);
    prod_m.compute_at(q, Var::outermost());
    p_m.compute_at(q, Var::outermost());

    inv.compute_root().reorder(c, x, y);
    inv_.compute_at(inv, x);
    s_m.compute_root();
    sigma.compute_at(inv, x);

    mu.compute_root();
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
    cost_right(x, y, d) = (1 - alpha) * clamp((diff(x+d, y, 0, d) + diff(x+d, y, 1, d) + diff(x+d, y, 2, d))/3, 0, threshColor) +
                          alpha * clamp(abs(left_gradient(x+d, y) - right_gradient(x, y)), 0, threshGrad);

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

    /***************** default schedule *******************/
    disp.compute_root();
    disp_left.compute_root()
             .update().reorder(x, y, rd);
    disp_right.compute_root()
              .update().reorder(x, y, rd);

    filtered_left.compute_at(disp_left, rd);
    filtered_right.compute_at(disp_right, rd);
    cost_left.compute_at(filtered_left, Var::outermost());
    cost_right.compute_at(filtered_right, Var::outermost());

    left_gradient.compute_root();
    right_gradient.compute_root();

    check_inline_schedule(disp);

    disp.compile_to_lowered_stmt("disp.html", {}, HTML);
    return disp;
}


/**************************************************************************************/
/******************** functions with schedule *****************************************/


/******************************* schedule mean *****************************/
/* return I_copy if copies I
* this is to enable schedule functions at I_copy
*/
void meanWithSchedule(Func I, Func meanI, int r, Var compute_level, bool c_innermost, bool scheduleI = true, bool unroll = true)
{
    float scale = 1.0f/(2*r+1)/(2*r+1);
    Var x("x"), y("y"), c(I.function().args()[2]), d("d");
    Func I_vsum("vsum");
    RDom k(-r, 2*r+1);
    int vector_width = 8;

    if (I.dimensions() == 3)
    {
        Func small_vsum("small_vsum"), small_hsum("small_hsum");
        small_vsum(x, y, c)= I(x, y, c) + I(x+1, y, c) + I(x+2, y, c) + I(x+3, y, c);
        RDom rt(0, (r+1)/2, "rt");
        if (r % 2 == 1)
        {
            I_vsum(x, y, c) = sum(small_vsum(x-r+rt*4, y, c)) - I(x+r+1, y, c);
        }
        else
        {
            I_vsum(x, y, c) = sum(small_vsum(x-r+rt*4, y, c)) + I(x+r, y, c);
        }

        meanI(x, y, c) = sum(I_vsum(x, y+k, c)) * scale;

        /*************** Schedule *******************/
        small_vsum.compute_at(I_vsum, y).vectorize(x, vector_width);
        I_vsum.compute_at(meanI, compute_level).vectorize(x, vector_width);
        if  (scheduleI)
        {
            I.compute_at(I_vsum, y).vectorize(x, vector_width);
        }
        if (c_innermost)
        {
            I_vsum.reorder(c, x, y);
            I.reorder(c, x, y);
        }
        if (unroll)
        {
            I_vsum.unroll(c);
            I.unroll(c);
        }
    }
    else if (I.dimensions() == 4)
    {
        Func small_vsum("small_vsum"), small_hsum("small_hsum");
        small_vsum(x, y, c, d)= I(x, y, c, d) + I(x+1, y, c, d) + I(x+2, y, c, d) + I(x+3, y, c, d);
        RDom rt(0, (r+1)/2, "rt");
        if (r % 2 == 1)
        {
            I_vsum(x, y, c, d) = sum(small_vsum(x-r+rt*4, y, c, d)) - I(x+r+1, y, c, d);
        }
        else
        {
            I_vsum(x, y, c, d) = sum(small_vsum(x-r+rt*4, y, c, d)) + I(x+r, y, c, d);
        }

        meanI(x, y, c, d) = sum(I_vsum(x, y+k, c, d)) * scale;

        /*************** Schedule *******************/
        small_vsum.compute_at(I_vsum, y).vectorize(x, vector_width);
        I_vsum.compute_at(meanI, compute_level).vectorize(x, vector_width);
        if  (scheduleI)
        {
            I.compute_at(I_vsum, y).vectorize(x, vector_width);
        }
        if (c_innermost)
        {
            I_vsum.reorder(c, x, y, d);
            I.reorder(c, x, y, d);
        }
        if (unroll)
        {
            I_vsum.unroll(c).unroll(d);
            I.unroll(c).unroll(d);
        }
    }
}

void guidanceImageProcessing(Func I, Func& meanI, Func& inv, int r, float eps)
{
    Var x("x"), y("y"), c("c"), d("d");
    float scale = 1.0f/(2*r+1)/(2*r+1);

    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    meanWithSchedule(I, meanI, r, Var::outermost(), false/*c_innermost*/, false/*scheduleI*/);

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

    inv_.compute_at(inv, xi).reorder(c, x, y).unroll(c).vectorize(x, vector_width);
    inv_.update().vectorize(x, vector_width);
    inv_.update(1).vectorize(x, vector_width);
    inv_.update(2).vectorize(x, vector_width);
    inv_.update(3).vectorize(x, vector_width);
    inv_.update(4).vectorize(x, vector_width);
    inv_.update(5).vectorize(x, vector_width);
    inv_.update(6).vectorize(x, vector_width);
    sigma.compute_at(inv, xi).reorder(c, x, y).unroll(c).vectorize(x, vector_width);
    s_m.compute_at(inv, xo).reorder(c, x, y).unroll(c).vectorize(x, vector_width);

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
    prod(x, y, c, d) = I(x, y, c, d) * p(x, y, d);

    Func prod_m("prod_m");
    meanWithSchedule(prod, prod_m, r, d, true);

    Func a_factor("a_factor");
    a_factor(x, y, c, d) = prod_m(x, y, c, d) - meanI(x, y, c, d) * meanP(x, y, d);

    Func ab("ab");
    RDom k(0, 3, "k");
    RDom rc(0, 4, "rc");
    ab(x, y, c, d) = undef<float>();
    ab(x, y, rc, d) = select(rc == 3,
                            meanP(x, y, d) - ab(x, y, 0, d) * meanI(x, y, 0, d)
                                           - ab(x, y, 1, d) * meanI(x, y, 1, d)
                                           - ab(x, y, 2, d) * meanI(x, y, 2, d),
                            inv(x, y, clamp(pair2ind(rc, 0), 0, 5), d) * a_factor(x, y, 0, d)
                          + inv(x, y, clamp(pair2ind(rc, 1), 0, 5), d) * a_factor(x, y, 1, d)
                          + inv(x, y, clamp(pair2ind(rc, 2), 0, 5), d) * a_factor(x, y, 2, d) );

    Func ab_m("ab_m");
    meanWithSchedule(ab, ab_m, r, d, true);

    filtered(x, y, d) = ab_m(x, y, 3, d)
                      + ab_m(x, y, 0, d) * I(x, y, 0, d)
                      + ab_m(x, y, 1, d) * I(x, y, 1, d)
                      + ab_m(x, y, 2, d) * I(x, y, 2, d);

    /************************** Schedule *************************/
    int vector_width = 8;
    ab_m.compute_at(filtered, d).reorder(x, y, c, d).unroll(c).unroll(d).vectorize(x, vector_width);
    ab.update().reorder(rc, x, y, d).unroll(rc).unroll(d).vectorize(x, vector_width);

    a_factor.compute_at(ab, x).reorder(c, x, y, d).unroll(c).unroll(d).vectorize(x, vector_width);

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

    Func mean_left_4d("mean_left_4d"), mean_right_4d("mean_right_4d"), inv_left_4d("inv_left_4d"), inv_right_4d("inv_right_4d");
    Func left_4d("left_4d"), right_4d("right_4d");
    left_4d(x, y, c, d) = left(x, y, c);
    right_4d(x, y, c, d) = right(x-d, y, c);
    mean_left_4d(x, y, c, d) = mean_left(x, y, c);
    mean_right_4d(x, y, c, d) = mean_right(x-d, y, c);
    inv_left_4d(x, y, c, d) = inv_left(x, y, c);
    inv_right_4d(x, y, c, d) = inv_right(x-d, y, c);

    Func mean_cost("mean_cost");
    meanWithSchedule(cost, mean_cost, r, d, false/*d innermost*/, false/*scheduleI*/);

    Func filtered_left("filtered_left"), filtered_right("filtered_right");
    filteringCost(left_4d, cost, mean_left_4d, mean_cost, inv_left_4d, filtered_left, r);
    filteringCost(right_4d, cost, mean_right_4d, mean_cost, inv_right_4d, filtered_right, r);

    RDom rd(0, numDisparities, "rd");
    Func disp_left_right("disp_left_right");
    disp_left_right(x, y) = {0, INFINITY, 0, INFINITY};
    Expr left_cond = filtered_left(x, y, rd) < disp_left_right(x, y)[1];
    Expr right_cond = filtered_right(x+rd, y, rd) < disp_left_right(x, y)[3];
    disp_left_right(x, y) = {
        select(left_cond, rd, disp_left_right(x, y)[0]),
        select(left_cond, filtered_left(x, y, rd), disp_left_right(x, y)[1]),
        select(right_cond, rd, disp_left_right(x, y)[2]),
        select(right_cond, filtered_right(x+rd, y, rd), disp_left_right(x, y)[3]),
    };

    Func disp("disp");
    Expr disp_val = disp_left_right(x, y)[0];
    disp(x, y) = select(x > disp_val && abs(disp_left_right(clamp(x - disp_val, 0, width-1), y)[2] - disp_val) < 1, disp_val, -1);

    /****************************** Schedule ****************************/
    int vector_width = 8;
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    disp.compute_root().vectorize(x, vector_width).parallel(x);
    disp_left_right.compute_root().vectorize(x, vector_width)
                   .update().tile(x, y, xo, yo, xi, yi, 32, 32).reorder(xi, yi, xo, yo, rd).vectorize(xi, vector_width);

    filtered_left.compute_at(disp_left_right, rd).tile(x, y, xo, yo, xi, yi, 128, 64)
                 .reorder(xi, yi, d, xo, yo).vectorize(xi, vector_width);
    filtered_right.compute_at(disp_left_right, rd).tile(x, y, xo, yo, xi, yi, 128, 64)
                 .reorder(xi, yi, d, xo, yo).vectorize(xi, vector_width);
    mean_cost.compute_at(disp_left_right, rd).vectorize(x, vector_width);
    cost.compute_at(disp_left_right, rd).vectorize(x, vector_width);

    mean_left.compute_root().vectorize(x, vector_width);
    inv_left.compute_root().reorder(c, x, y).vectorize(x, vector_width);
    mean_right.compute_root().vectorize(x, vector_width);
    inv_right.compute_root().reorder(c, x, y).vectorize(x, vector_width);

    left.compute_root().vectorize(x, vector_width);
    right.compute_root().vectorize(x, vector_width);
    left_gradient.compute_root().vectorize(x, vector_width);
    right_gradient.compute_root().vectorize(x, vector_width);
    return disp;
}

void guidedFilterTest()
{
    Image<float> im0 = Halide::Tools::load_image("../../trainingQ/Teddy/im0.png");
    Var x("x"), y("y"), c("c"), d("d");
    Var xi("xi"), xo("xo"), yi("yi"), yo("yo");

    // test meanWithSchedule
    {
        Func I3("I3"), I4("I4");
        I3(x, y, c) = x+y;
        I4(x, y, c, d) = x+y;

        Func output3("output3"), output4("output4");
        meanWithSchedule(I3, output3, 9, xo, true, true/*scheduleI*/, false/*unroll*/);
        meanWithSchedule(I4, output4, 9, xo, true, true/*scheduleI*/, false/*unroll*/);

        output3.compute_root().tile(x, y, xo, yo, xi, yi, 128, 64).reorder(c, xi, yi, xo, yo).vectorize(xi, 8);
        output4.compute_root().tile(x, y, xo, yo, xi, yi, 128, 64).reorder(c, xi, yi, xo, yo, d).vectorize(xi, 8);
        int w = 600, h = 400;
        Image<float> image3 = output3.realize(w, h, 3);
        Image<float> image4 = output4.realize(w, h, 3, 2);

        for (int x = 9; x < w-9; x++)
        {
            for (int y = 9; y < h-9; y++)
            {
                for (int c = 0; c < 3; c++)
                {
                    assert(std::abs(image3(x, y, c)- x - y) < 0.01);
                    assert(std::abs(image4(x, y, c, 0) - x - y) < 0.01);
                    assert(std::abs(image4(x, y, c, 0) - x - y) < 0.01);
                }
            }
        }
    }

    {
        Func I("I"),  meanI("meanI");
        I(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
        meanWithSchedule(I, meanI, 9, xo, true, true/*scheduleI*/, false/*unroll*/);
        meanI.compute_root().tile(x, y, xo, yo, xi, yi, 128, 64).reorder(c, xi, yi, xo, yo).vectorize(xi, 8);

        meanI.compile_to_lowered_stmt("mean.html", {}, HTML);
        Target t = get_jit_target_from_environment().with_feature(Target::Profile);
        meanI.realize(im0.width(), im0.height(), 3, t);
        printf("mean with schedule");
        profile(meanI, im0.width(), im0.height(), 3, 5000);
    }

    {
        Func I("I");
        I(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
        Func meanI = mean(I, 9);
        meanI.compute_root();

        meanI.compile_to_lowered_stmt("mean.html", {}, HTML);
        Target t = get_jit_target_from_environment().with_feature(Target::Profile);
        meanI.realize(im0.width(), im0.height(), 3, t);
        printf("mean without schedule");
        profile(meanI, im0.width(), im0.height(), 3, 5000);
    }

    // {
    //     Image<float> im1 = Halide::Tools::load_image("../../trainingQ/Teddy/im1.png");
    //     Func I("I"), I1("I1"), meanI("meanI"), inv("inv");
    //     I(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    //     I1(x, y, c) = im1(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
    //     guidanceImageProcessing(I, meanI, inv, 9, 0.01);
    //
    //     int vector_width = 8;
    //     Var xi("xi"), xo("xo"), yi("yi"), yo("yo");
    //     inv.compute_root().tile(x, y, xo, yo, xi, yi, 32, 32).reorder(c, xi, yi, xo, yo).vectorize(xi, vector_width);
    //     I.compute_root().vectorize(x, vector_width);
    //     I1.compute_root().vectorize(x, vector_width);
    //     meanI.compute_at(inv, xo).vectorize(x, vector_width);
    //
    //     Target t = get_jit_target_from_environment().with_feature(Target::Profile);
    //     inv.realize(im0.width(), im0.height(), 6, t);
    //     inv.compile_to_lowered_stmt("filter.html", {}, HTML);
    //     profile(inv, im0.width(), im0.height(), 6, 10);
    //
    // }

    {
        // Func I_d("I_d"), meanI_d("meanI_d"), inv_d("inv_d");
        // I_d(x, y, c, d) = I(x, y, c);
        // meanI_d(x, y, c, d) = I(x, y, c);
        // inv_d(x, y, c, d) = 1.0f;
        //
        // Func cost("cost"), meanCost("meanCost");
        // cost(x, y, d) = abs(I(x, y, 0) - I1(x, y, 0)) + abs(I(x, y, 1) - I1(x, y, 1)) + abs(I(x, y, 2) - I1(x, y, 2));
        // meanWithSchedule(cost, meanCost, 9, d, false, false);
        //
        // Func filtered;
        // filteringCost(I_d, cost, meanI_d, meanCost, inv_d, filtered, 9);
        //
        // int vector_width = 8;
        // filtered.compute_root().vectorize(x, vector_width);
        // I.compute_at(filtered, d).vectorize(x, vector_width);
        // meanI.compute_at(filtered, d).vectorize(x, vector_width);
        // inv.compute_at(filtered, d).reorder(c, x, y).vectorize(x, vector_width);
        // cost.compute_at(filtered, d).vectorize(x, vector_width);
        // meanCost.compute_at(filtered, d).vectorize(x, vector_width);
        //
        // filtered.compile_to_lowered_stmt("filter.html", {}, HTML);
        // Target t = get_jit_target_from_environment().with_feature(Target::Profile);
        // filtered.realize(im0.width(), im0.height(), 1, t);
        // profile(filtered, im0.width(), im0.height(), 1, 50);

    }

    {
        //
        // Image<float> im1 = Halide::Tools::load_image("../../trainingQ/Teddy/im1.png");
        // Func left("left"), right("right");
        // left(x, y, c) = im0(clamp(x, 0, im0.width()-1), clamp(y, 0, im0.height()-1), c);
        // right(x, y, c) = im1(clamp(x, 0, im1.width()-1), clamp(y, 0, im1.height()-1), c);
        // Func disp = stereoGF_scheduled(left, right, im0.width(), im0.height(), 9, 0.01, 1, 0.9, 0.0028, 0.008);
        //
        // Target t = get_jit_target_from_environment().with_feature(Target::Profile);
        // disp.compile_to_lowered_stmt("disp.html", {}, HTML, t);
        // Image<int> disp_image = disp.realize(im0.width(), im0.height(), t);
        //
        // Image<float> scaled_disp(im0.width(), im0.height());
        // for (int y = 0; y < disp_image.height(); y++) {
        //     for (int x = 0; x < disp_image.width(); x++) {
        //         scaled_disp(x, y) = std::min(1.f, std::max(0.f, disp_image(x,y) * 1.0f / 59));
        //     }
        // };
        //
        // Halide::Tools::save_image(scaled_disp, "disp.png");
        // profile(disp, im0.width(), im0.height(), 10);
    }
}

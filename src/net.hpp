#pragma once
#define VEC_OPTIMIZE
//#define VEC_DEBUG
#define MAT_OPTIMIZE
//#define MAT_DEBUG
#include "../CUtil/util.hpp"
#include <cmath>

typedef double (*activation_func_t)(double);

struct NNet{
    const vec<size_t> shape;
    dmat** weights;
    dvec** biases;
    activation_func_t activation_function;

    NNet(vec<size_t> shape,activation_func_t act_func);
    ~NNet();
    dvec eval(dvec in);
    void mutate(double std_dev);
    NNet* clone();
    void copy(NNet& b);

    static struct Activation{
        static double logistic(double x);
        static double sigmoid(double x);
        static double relu(double x);
        static double dying_sigmoid(double x);
        static double softplus(double x);
        static double swish(double x);
    } activation;
};

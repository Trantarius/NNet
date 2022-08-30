#pragma once
#include "../CUtil/util.hpp"

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
};

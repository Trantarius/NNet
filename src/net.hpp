#pragma once
#include "../CUtil/util.hpp"

struct NNet{
    const vec<size_t> shape;
    dmat** weights;
    dvec** biases;
    double (*activation_function)(double);

    NNet(vec<size_t> shape,double (*afunc)(double));
    ~NNet();
    dvec eval(dvec in);
    void mutate(double std_dev);
};

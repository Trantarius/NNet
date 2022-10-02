#pragma once
#include "net.hpp"
#include <cmath>
#include <algorithm>

struct NetEntry{
    NNet* net;
    double performance=(double)(uint)(-1);
    NetEntry(NNet* net=NULL,double performance=(double)(uint)(-1)):net(net),performance(performance){}
};

class Trainer{
public:
    const vec<size_t> shape;
    const activation_func_t act_func;

    Trainer(vec<size_t> shape,activation_func_t activ):
        shape(shape),act_func(activ){}

    virtual NetEntry train()=0;
};

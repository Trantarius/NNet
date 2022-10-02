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
    const Netshape net_shape;

    Trainer(Netshape shape):net_shape(shape){}

    virtual NetEntry train()=0;
};

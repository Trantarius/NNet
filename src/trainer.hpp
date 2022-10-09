#pragma once
#include "net.hpp"
#include <cmath>
#include <algorithm>


class Trainer{
public:
    //determines the parameters of the network(s) to be trained
    const Netshape net_shape;

    Trainer(Netshape shape):net_shape(shape){}

    virtual NNet train()=0;
};

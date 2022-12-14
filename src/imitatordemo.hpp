#pragma once
#include <iostream>
#include "net.hpp"
#include "imitator.hpp"
#include <cmath>
#include <ctime>


namespace ImitatorDemo{

    dvec target_function(dvec in);

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NNet* net);

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NNet* net);

    void demo();
    void demo_backprop();
};

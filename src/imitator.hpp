#pragma once
#include "net.hpp"
#include "montecarlo.hpp"

class ImitatorTrainer:public MonteCarloTrainer{
public:
    size_t samples_per_net=100;
    dvec (*target_function)(dvec);
    virtual double perform(const NNet& net);

    ImitatorTrainer(Netshape shape,dvec(*target_func)(dvec)):
        MonteCarloTrainer(shape),target_function(target_func){}
};

#pragma once
#include "net.hpp"
#include "montecarlo.hpp"

class ImitatorTrainer:public MonteCarloTrainer{
public:
    size_t samples_per_net=100;
    dvec (*target_function)(dvec);
    virtual double perform(const NNet& net);

    ImitatorTrainer(vec<size_t> shape,activation_func_t act_func,dvec(*target_func)(dvec)):
        MonteCarloTrainer(shape,act_func),target_function(target_func){}
};

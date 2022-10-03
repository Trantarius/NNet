#pragma once
#include "net.hpp"
#include "montecarlo.hpp"

/*
 * Trains a network to imitate a known function. Always trains on the range [0,1).
 * Networks are evaluated against the function directly for randomly generated input
 * samples.
 */
class ImitatorTrainer:public MonteCarloTrainer{
public:
    //number of randomly generated inputs to test each network on
    size_t samples_per_net=100;
    //the function to be imitated. must have same input/output size as the network
    dvec (*target_function)(dvec);
    virtual double perform(const NNet& net);

    ImitatorTrainer(Netshape shape,dvec(*target_func)(dvec)):
        MonteCarloTrainer(shape),target_function(target_func){}
};

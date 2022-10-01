#pragma once
#include "net.hpp"
#include "trainer.hpp"

class ImitatorTrainer:public Trainer{
public:
    size_t samples_per_net=100;
    dvec (*target_function)(dvec);
    virtual double perform(NNet& net);

    ImitatorTrainer(vec<size_t> shape,activation_func_t act_func,dvec(*target_func)(dvec)):
        Trainer(shape,act_func),target_function(target_func){}
};

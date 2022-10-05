#pragma once
#include "trainer.hpp"

/*
 * Calculate gradient of a network for a specific input and desired output.
 * see https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
 * also see https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication
 */
NNet* backprop(const NNet& net,dvec in,dvec expected);
//adds net_b to net_a (ie, adds all their weights and biases), with net_b elements multiplied
//by 'scalar'. useful for operations involving a gradient NNet.
void net_add(NNet* net_a,const NNet* net_b,double scalar);

struct BackPropTrainer:public Trainer{

    typedef std::pair<dvec,dvec> Sample;
    typedef void(*callback_t)(BackPropTrainer*,size_t,NNet*);

    size_t samples_per_gen=100;
    size_t gen_count=100;
    double learn_rate=0.01;

    //params are this trainer, index of the generation, and the best network this generation
    callback_t gen_callback=[](BackPropTrainer*,size_t,NNet*){};

    BackPropTrainer(Netshape shape):Trainer(shape){}

    NNet* train();
    virtual Sample sample(size_t n)=0;
};

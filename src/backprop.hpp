#pragma once
#include "trainer.hpp"

NNet backprop(NNet& net,dvec in,dvec expected);

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

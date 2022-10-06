#pragma once
#include "trainer.hpp"

/*
 * Trainer that uses monte carlo (ie random mutation) method. Each subsequent generation
 * is biased to be more similar to the better networks of the previous generation.
 * Needs a definition for perform() to be used, allowing supervised and
 * unsupervised training.
 */
class MonteCarloTrainer:public Trainer{
    Threadpool threadpool;
public:
    enum SORT_MODE{ASCENDING,DESCENDING} sort_mode=ASCENDING;

    typedef void(*callback_t)(MonteCarloTrainer*,size_t,NNet*);

    //called whenever a generation is completed
    //params are this trainer, index of the generation, and the best network this generation
    callback_t gen_callback=[](MonteCarloTrainer*,size_t,NNet*){};
    //called whenever a network has its performance calculated
    //params are this trainer, number of perfs completed this generation, and the newly calculated NetEntry
    callback_t perf_callback=[](MonteCarloTrainer*,size_t,NNet*){};

    //number of generations to train when train() is called.
    size_t gen_count=100;
    //number of networks in each generation
    size_t nets_per_gen=1000;
    //the standard deviation of the change of each parameter of each network between generations
    double mutation_rate=0.01;
    //multiplies the mutation rate every generation for a primitive adaptive learning rate
    double mutation_adapt=1.0;
    //determines how strongly to bias the next generation to be descended from the best of the last generation
    //reasonable range is [0.5,inf); 0.5 is no bias
    //at 8,  ~50% come from the best 10%
    //at 14, ~80% come from the best 10%
    //at 20, ~90% come from the best 10%
    size_t keep_ratio=10;
    //logs a csv with all of the performances of each generation while training
    bool log_enabled=false;
    string logpath="log.csv";

    MonteCarloTrainer(Netshape shape):
        Trainer(shape),threadpool(16){}

    virtual double perform(const NNet& net)=0;

    NNet* train();

    Threadpool* get_threadpool(){return &threadpool;}
};

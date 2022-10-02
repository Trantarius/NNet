#pragma once
#include "trainer.hpp"

class MonteCarloTrainer:public Trainer{
    Threadpool threadpool;
public:
    enum SORT_MODE{ASCENDING,DESCENDING} sort_mode=ASCENDING;

    typedef void(*callback_t)(MonteCarloTrainer*,size_t,NetEntry);

    callback_t gen_callback=[](MonteCarloTrainer*,size_t,NetEntry){};
    callback_t perf_callback=[](MonteCarloTrainer*,size_t,NetEntry){};

    size_t gen_count=100;
    size_t nets_per_gen=1000;
    double mutation_rate=0.01;
    size_t keep_ratio=10;
    bool log_enabled=false;
    string logpath="log.csv";

    MonteCarloTrainer(Netshape shape):
        Trainer(shape),threadpool(16){}

    virtual double perform(const NNet& net)=0;
    void generation(vec<NetEntry>& last,vec<NetEntry>& next);

    NetEntry train();

    Threadpool* get_threadpool(){return &threadpool;}
};

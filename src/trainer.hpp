#pragma once
#include "net.hpp"
#include <cmath>
#include <algorithm>

struct NetEntry{
    NNet* net;
    double performance=(double)(uint)(-1);
    NetEntry(NNet* net=NULL,double performance=(double)(uint)(-1)):net(net),performance(performance){}
};

class Trainer{
    Threadpool threadpool;
public:
    enum SORT_MODE{ASCENDING,DESCENDING} sort_mode=ASCENDING;
    const vec<size_t> shape;
    const activation_func_t act_func;

    typedef void(*callback_t)(Trainer*,size_t,NetEntry);

    callback_t gen_callback=[](Trainer*,size_t,NetEntry){};
    callback_t perf_callback=[](Trainer*,size_t,NetEntry){};

    size_t nets_per_gen=1000;
    double mutation_rate=0.01;
    size_t keep_ratio=10;
    bool log_enabled=false;
    string logpath="log.csv";

    Trainer(vec<size_t> shape,activation_func_t activ):
        shape(shape),act_func(activ),threadpool(16){}

    virtual double perform(const NNet& net)=0;
    void generation(vec<NetEntry>& last,vec<NetEntry>& next);

    NetEntry train(size_t gen_count);

    Threadpool* get_threadpool(){return &threadpool;}
};

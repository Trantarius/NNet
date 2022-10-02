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
public:
    enum SORT_MODE{ASCENDING,DESCENDING} sort_mode=ASCENDING;
    const vec<size_t> shape;
    const activation_func_t act_func;

    void (*gen_callback)(Trainer* trainer,size_t i,NetEntry entry)=[](Trainer*,size_t,NetEntry){};
    void (*perf_callback)(Trainer* trainer,size_t i,NetEntry entry)=[](Trainer*,size_t,NetEntry){};

    size_t nets_per_gen=1000;
    double mutation_rate=0.01;
    size_t keep_ratio=10;

    Trainer(vec<size_t> shape,activation_func_t activ):
        shape(shape),act_func(activ){}

    virtual double perform(const NNet& net)=0;
    void generation(vec<NetEntry>& last,vec<NetEntry>& next);

    NetEntry train(size_t gen_count);
};

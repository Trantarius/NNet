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
    enum SORT_MODE{ASCENDING,DESCENDING} sort_mode;
    const vec<size_t> shape;
    const activation_func_t act_func;

    double (*perform_func)(NNet&);
    void (*gen_callback)(NetEntry entry)=[](NetEntry){};

    size_t nets_per_gen=10000;
    double mutation_rate=0.01;
    size_t keep_ratio=10;

    Trainer(vec<size_t> shape,activation_func_t activ,double(*perform_func)(NNet&)):
        shape(shape),act_func(activ),perform_func(perform_func){}

    void generation(vec<NetEntry>& last,vec<NetEntry>& next);

    NetEntry train(size_t gen_count);
};

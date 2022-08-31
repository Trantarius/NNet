#pragma once
#include "net.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

struct NetEntry{
    NNet* net;
    double performance=(double)(uint)(-1);
    NetEntry(NNet* net=NULL,double performance=(double)(uint)(-1)):net(net),performance(performance){}
};

struct Trainer{
    const vec<size_t> shape;
    const activation_func_t act_func;

    static bool sort_ascending(const NetEntry& a,const NetEntry& b){
        return a.performance<b.performance;
    }
    static bool sort_descending(const NetEntry& a,const NetEntry& b){
        return a.performance<b.performance;
    }
    bool(*sort_mode)(const NetEntry&,const NetEntry&)=sort_ascending;

    double (*perform_func)(NNet&);
    void (*gen_callback)(NetEntry entry)=[](NetEntry){};

    size_t nets_per_gen=10000;
    size_t samples_per_net=100;
    double mutation_rate=0.01;
    size_t keep_ratio=10;

    Trainer(vec<size_t> shape,activation_func_t activ,double(*perform_func)(NNet&)):
        shape(shape),act_func(activ),perform_func(perform_func){}

    void generation(vec<NetEntry>& last,vec<NetEntry>& next);

    NetEntry train(size_t gen_count);
};

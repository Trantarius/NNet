#pragma once
#include "net.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

struct Trainer{
    const vec<size_t> shape;
    double (* const act_func)(double);
    dvec (* const target_func)(dvec);

    size_t nets_per_gen=10000;
    size_t samples_per_net=100;
    double mutation_rate=0.01;
    size_t keep_ratio=10;

    Trainer(vec<size_t>& shape,double (*activ)(double),dvec (*target)(dvec)):
        shape(shape),act_func(activ),target_func(target){}

    struct NetEntry{
        NNet* net;
        double performance=(double)(uint)(-1);
        bool operator<(const NetEntry& b){
            return performance<b.performance;
        }
        NetEntry():net(NULL){}
        NetEntry(NNet* net):net(net){}
    };

    dvec randvec(size_t s);

    double perform(NetEntry& ne);

    void generation(std::vector<NetEntry>& last,std::vector<NetEntry>& next);

    NNet* train(size_t gen_count,vec<size_t> shape,double(*act_func)(double));
};

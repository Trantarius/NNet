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

    dvec randvec(size_t s){
        dvec v(s);
        for(int n=0;n<s;n++){
            v[n]=(double)rand()/RAND_MAX;
        }
        return v;
    }

    double perform(NetEntry& ne){
        dvec* inputs=dvec::new_array(shape[0],samples_per_net);
        dvec* target=dvec::new_array(shape[shape.size()-1],samples_per_net);
        dvec* output=dvec::new_array(shape[shape.size()-1],samples_per_net);
        double total_err=0;
        for(size_t n=0;n<samples_per_net;n++){
            inputs[n]=randvec(shape[0]);
            target[n]=target_func(inputs[n]);
            output[n]=ne.net->eval(inputs[n]);
            dvec errv=output[n]-target[n];
            for(int i=0;i<errv.size();i++){
                total_err+=fabs(errv[i]);
            }
        }
        dvec::delete_array(inputs);
        dvec::delete_array(target);
        dvec::delete_array(output);
        return total_err;
    }

    void generation(std::vector<NetEntry>& last,std::vector<NetEntry>& next){
        for(size_t n=0;n<nets_per_gen;n++){
            next[n].net->copy(*last[n/keep_ratio].net);
            next[n].net->mutate(mutation_rate);
            next[n].performance=perform(next[n]);
        }
        sort(next.begin(),next.end());
    }

    NNet* train(size_t gen_count,vec<size_t> shape,double(*act_func)(double)){
        std::vector<NetEntry> gen(nets_per_gen);
        std::vector<NetEntry> alt(nets_per_gen);
        for(size_t n=0;n<nets_per_gen;n++){
            gen[n]=NetEntry(new NNet(shape,act_func));
            alt[n]=NetEntry(new NNet(shape,act_func));
        }
        for(size_t n=0;n<gen_count;n++){
            generation(gen,alt);
            swap(gen,alt);
            print(gen[0].performance);
        }
        NNet* ret=gen[0].net->clone();
        for(size_t n=0;n<gen.size();n++){
            delete gen[n].net;
            delete alt[n].net;
        }
        return ret;
    }
};

#pragma once
#include "net.hpp"
#include <cmath>
#include <list>

struct Trainer{
    size_t nets_per_gen=10000;
    size_t samples_per_net=100;
    double mutation_rate=0.01;
    size_t keep_ratio=10;

    dvec target_function(dvec in){
        return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
    }

    struct NetEntry{
        NNet* net;
        double performance=(double)(uint)(-1);
        bool operator<(const NetEntry& b){
            return performance<b.performance;
        }
        NetEntry(NNet* net):net(net){}
    };

    dvec rand_5(){
        dvec v(5);
        for(int n=0;n<5;n++){
            v[n]=(double)rand()/RAND_MAX;
        }
        return v;
    }

    double perform(NetEntry& ne){
        dvec* inputs=dvec::new_array(5,samples_per_net);
        dvec* target=dvec::new_array(5,samples_per_net);
        dvec* output=dvec::new_array(5,samples_per_net);
        double total_err=0;
        for(size_t n=0;n<samples_per_net;n++){
            inputs[n]=rand_5();
            target[n]=target_function(inputs[n]);
            output[n]=ne.net->eval(inputs[n]);
            dvec errv=output[n]-target[n];
            for(int i=0;i<5;i++){
                total_err+=fabs(errv[i]);
            }
        }
        dvec::delete_array(inputs);
        dvec::delete_array(target);
        dvec::delete_array(output);
        return total_err;
    }

    std::list<NetEntry> generation(std::list<NetEntry> last){
        std::list<NetEntry> next;
        auto last_it=last.begin();
        for(size_t n=0;n<nets_per_gen;n++){
            next.push_back(NetEntry((*last_it).net->clone()));
            next.back().net->mutate(mutation_rate);
            next.back().performance=perform(next.back());
            if(n%keep_ratio==keep_ratio-1){
                last_it++;
            }
        }
        next.sort();
        return next;
    }

    NNet* train(size_t gen_count,vec<size_t> shape,double(*act_func)(double)){
        std::list<NetEntry> gen;
        for(size_t n=0;n<nets_per_gen;n++){
            gen.push_back(NetEntry(new NNet(shape,act_func)));
        }
        std::list<NetEntry> tmp;
        for(size_t n=0;n<gen_count;n++){
            tmp.swap(gen);
            gen=generation(tmp);
            for(auto tmp_it=tmp.begin();tmp_it!=tmp.end();tmp_it++){
                delete (*tmp_it).net;
            }
            print(gen.front().performance);
        }
        auto gen_it=gen.begin();
        gen_it++;
        for(;gen_it!=gen.end();gen_it++){
                delete (*gen_it).net;
            }
        return gen.front().net;
    }
};

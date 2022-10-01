#include "trainer.hpp"

double randf(){
    return (double)rand()/RAND_MAX;
}

size_t biased_idx(size_t base,size_t bias_strength,size_t max){
    //return base/bias_strength;
    double a=bias_strength;
    double x=(double)base/max;
    x=(a*pow(x,2*a)+2*x)/(2+a);
    return (size_t)(x*max)%max;
}

bool sort_ascending_comp(const NetEntry& a,const NetEntry& b){
    return a.performance<b.performance;
}
bool sort_descending_comp(const NetEntry& a,const NetEntry& b){
    return b.performance<a.performance;
}

void Trainer::generation(vec<NetEntry>& last,vec<NetEntry>& next){
    for(size_t n=0;n<nets_per_gen;n++){
        next[n].net->copy(*last[biased_idx(n,keep_ratio,nets_per_gen)].net);
        next[n].net->mutate(mutation_rate);
        next[n].performance=perform(*next[n].net);
    }

    std::sort(&next,&next+next.size(),(
        sort_mode==SORT_MODE::ASCENDING?
            sort_ascending_comp:
            sort_descending_comp
    ));
}

NetEntry Trainer::train(size_t gen_count){
    vec<NetEntry> gen(nets_per_gen);
    vec<NetEntry> alt(nets_per_gen);
    for(size_t n=0;n<nets_per_gen;n++){
        gen[n]=NetEntry(new NNet(shape,act_func));
        alt[n]=NetEntry(new NNet(shape,act_func));
    }
    for(size_t n=0;n<gen_count;n++){
        generation(gen,alt);
        swap(gen,alt);
        gen_callback(gen[0]);
    }
    NetEntry ret=NetEntry(gen[0].net->clone(),gen[0].performance);
    for(size_t n=0;n<gen.size();n++){
        delete gen[n].net;
        delete alt[n].net;
    }
    return ret;
}

#include "trainer.hpp"



void Trainer::generation(std::vector<NetEntry>& last,std::vector<NetEntry>& next){
    for(size_t n=0;n<nets_per_gen;n++){
        next[n].net->copy(*last[n/keep_ratio].net);
        next[n].net->mutate(mutation_rate);
        next[n].performance=perform_func(*next[n].net);
    }
    sort(next.begin(),next.end(),sort_mode);
}

NetEntry Trainer::train(size_t gen_count){
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
    NetEntry ret=NetEntry(gen[0].net->clone(),gen[0].performance);
    for(size_t n=0;n<gen.size();n++){
        delete gen[n].net;
        delete alt[n].net;
    }
    return ret;
}

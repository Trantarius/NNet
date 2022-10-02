#include "montecarlo.hpp"
#include <fstream>

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

struct NetTask:public Task{
    size_t n=-1;
    MonteCarloTrainer* trainer;
    vec<NetEntry>* last;
    vec<NetEntry>* next;
    void perform(){
        (*next)[n].net->copy(*(*last)[biased_idx(n,trainer->keep_ratio,trainer->nets_per_gen)].net);
        (*next)[n].net->mutate(trainer->mutation_rate);
        (*next)[n].performance=trainer->perform(*(*next)[n].net);
        trainer->perf_callback(trainer,trainer->nets_per_gen-trainer->get_threadpool()->tasks_left(),(*next)[n]);
    }
    NetTask(vec<NetEntry>* last,vec<NetEntry>* next,MonteCarloTrainer* trainer,size_t n):
    last(last),next(next),trainer(trainer),n(n){}
};

void MonteCarloTrainer::generation(vec<NetEntry>& last,vec<NetEntry>& next){
    for(size_t n=0;n<nets_per_gen;n++){
        threadpool.push(new NetTask(&last,&next,this,n));
        /*
         *        next[n].net->copy(*last[biased_idx(n,keep_ratio,nets_per_gen)].net);
         *        next[n].net->mutate(mutation_rate);
         *        next[n].performance=perform(*next[n].net);
         *        perf_callback(this,n,next[n]);
         */
    }
    threadpool.finish();

    std::sort(next.ptr(),next.ptr()+next.size(),(
        sort_mode==SORT_MODE::ASCENDING?
        sort_ascending_comp:
        sort_descending_comp
    ));
}

NetEntry MonteCarloTrainer::train(){
    vec<NetEntry> gen(nets_per_gen);
    vec<NetEntry> alt(nets_per_gen);
    for(size_t n=0;n<nets_per_gen;n++){
        gen[n]=NetEntry(new NNet(net_shape));
        alt[n]=NetEntry(new NNet(net_shape));
    }

    std::fstream logfile;
    if(log_enabled){
        logfile.open(logpath,std::fstream::out);
    }

    for(size_t n=0;n<gen_count;n++){
        generation(gen,alt);
        swap(gen,alt);
        gen_callback(this,n,gen[0]);
        if(log_enabled){
            for(size_t l=0;l<gen.size();l++){
                logfile<<gen[l].performance<<',';
            }
            logfile<<std::endl;
        }
    }
    if(log_enabled){
        logfile.close();
    }
    NetEntry ret=NetEntry(gen[0].net->clone(),gen[0].performance);
    for(size_t n=0;n<gen.size();n++){
        delete gen[n].net;
        delete alt[n].net;
    }
    return ret;
}


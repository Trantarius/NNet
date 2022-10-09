#include "montecarlo.hpp"
#include <fstream>


//takes indices evenly distibuted in [0,max] and returns indices biased towards 0 in [0,max]
size_t biased_idx(size_t base,size_t bias_strength,size_t max){
    //return base/bias_strength;
    double a=bias_strength;
    double x=(double)base/max;
    x=(a*pow(x,2*a)+2*x)/(2+a);
    return (size_t)(x*max)%max;
}

bool sort_ascending_comp(NNet*const& a,NNet*const& b){
    return a->performance<b->performance;
}
bool sort_descending_comp(NNet*const& a,NNet*const& b){
    return b->performance<a->performance;
}

//used to create one network in a generation in a multithreaded manner
struct NetTask:public Task{
    size_t n=-1;
    MonteCarloTrainer* trainer;
    vec<NNet*>* last;
    vec<NNet*>* next;
    void perform(){
        //pick a network from the last generation, biased towards the ones with better
        //performance. that network is this ones "parent". copy it.
        (*next)[n]->copy(*(*last)[biased_idx(n,trainer->keep_ratio,trainer->nets_per_gen)]);
        //change a little
        (*next)[n]->mutate(trainer->mutation_rate);
        //test the new network
        (*next)[n]->performance=trainer->perform(*(*next)[n]);
        trainer->perf_callback(trainer,trainer->nets_per_gen-trainer->get_threadpool()->tasks_left(),(*next)[n]);
    }
    NetTask(vec<NNet*>* last,vec<NNet*>* next,MonteCarloTrainer* trainer,size_t n):
    last(last),next(next),trainer(trainer),n(n){}
};
NNet MonteCarloTrainer::train(){

    //make 2 generations of networks. we will train back and forth between these generations to
    //avoid reallocating a whole generation of nets.
    vec<NNet*> gen(nets_per_gen);
    vec<NNet*> alt(nets_per_gen);
    for(size_t n=0;n<nets_per_gen;n++){
        gen[n]=new NNet(net_shape);
        gen[n]->mutate(1.0);
        alt[n]=new NNet(net_shape);
    }

    std::sort(alt.ptr(),alt.ptr()+alt.size(),(
        sort_mode==SORT_MODE::ASCENDING?
        sort_ascending_comp:
        sort_descending_comp
    ));

    std::fstream logfile;
    if(log_enabled){
        logfile.open(logpath,std::fstream::out);
    }

    for(size_t n=0;n<gen_count;n++){

        for(size_t n=0;n<nets_per_gen;n++){
            //makes one network in the next generation
            //you could put the contents of NetTask::perform() here to do it single threaded
            threadpool.push(new NetTask(&gen,&alt,this,n));
        }
        threadpool.finish();

        //sort by performance so that better networks are at lower indices, best at 0
        std::sort(alt.ptr(),alt.ptr()+alt.size(),(
            sort_mode==SORT_MODE::ASCENDING?
            sort_ascending_comp:
            sort_descending_comp
        ));

        mutation_rate*=mutation_adapt;

        swap(gen,alt);
        gen_callback(this,n,gen[0]);
        if(log_enabled){
            for(size_t l=0;l<gen.size();l++){
                logfile<<gen[l]->performance<<',';
            }
            logfile<<std::endl;
        }
    }
    if(log_enabled){
        logfile.close();
    }

    //delete all networks, except the best one, which is returned
    NNet* ret=gen[0];
    delete alt[0];
    for(size_t n=1;n<gen.size();n++){
        delete gen[n];
        delete alt[n];
    }
    NNet net=*ret;
    delete ret;
    return net;
}


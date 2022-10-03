#include "imitatordemo.hpp"

namespace ImitatorDemo{

    dvec target_function(dvec in){
        return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
    }

    Timer timer;

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
        double t=timer.stop();
        printw(16,Timer::format(t),net->performance," "," "," "," ");
        timer.start();
    }

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
        static std::mutex mtx;
        mtx.lock();
        print_loadbar((double)n/trainer->nets_per_gen);
        mtx.unlock();
    }

    void demo(){
        srand(time(NULL));
        Netshape netshape(vec<size_t>(5,5),Activation::tanh);

        ImitatorTrainer trainer(netshape,target_function);
        trainer.gen_callback=gen_callback;
        trainer.perf_callback=perf_callback;

        trainer.gen_count=100;
        trainer.samples_per_net=10;
        trainer.nets_per_gen=10000;
        trainer.mutation_rate=0.05;
        trainer.keep_ratio=10;
        trainer.log_enabled=true;

        timer.start();
        trainer.train();
    }

};

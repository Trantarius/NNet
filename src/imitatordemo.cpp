#include "imitatordemo.hpp"

namespace ImitatorDemo{

    dvec target_function(dvec in){
        return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
    }

    Timer timer;

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NetEntry entry){
        double t=timer.stop();
        printw(16,Timer::format(t),entry.performance," "," "," "," ");
        timer.start();
    }

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NetEntry entry){
        static std::mutex mtx;
        mtx.lock();
        print_loadbar((double)n/trainer->nets_per_gen);
        mtx.unlock();
    }

    void demo(){
        srand(time(NULL));
        vec<size_t> netshape(5,5);

        ImitatorTrainer trainer(netshape,NNet::Activation::dying_sigmoid,target_function);
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

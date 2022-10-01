#include <iostream>
#include "net.hpp"
#include "imitator.hpp"
#include <cmath>
#include <ctime>

dvec target_function(dvec in){
    return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
}

Timer timer;

void callback(NetEntry entry){
    double t=timer.stop();
    print(Timer::format(t),"\t",entry.performance);
    timer.start();
}


int main(){
    srand(time(NULL));
    vec<size_t> netshape(5,5);

    ImitatorTrainer trainer(netshape,NNet::Activation::dying_sigmoid,target_function);
    trainer.gen_callback=callback;
    trainer.keep_ratio=10;
    timer.start();
    trainer.train(100);
}

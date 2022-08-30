#include <iostream>
#include "net.hpp"
#include "trainer.hpp"
#include <cmath>
#include <ctime>

dvec target_function(dvec in){
    return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
}

dvec randvec(size_t s){
    dvec v(s);
    for(int n=0;n<s;n++){
        v[n]=(double)rand()/RAND_MAX;
    }
    return v;
}

const size_t samples_per_net=100;

double perform(NNet& net){
    dvec* inputs=dvec::new_array(net.shape[0],samples_per_net);
    dvec* target=dvec::new_array(net.shape[net.shape.size()-1],samples_per_net);
    dvec* output=dvec::new_array(net.shape[net.shape.size()-1],samples_per_net);
    double total_err=0;
    for(size_t n=0;n<samples_per_net;n++){
        inputs[n]=randvec(net.shape[0]);
        target[n]=target_function(inputs[n]);
        output[n]=net.eval(inputs[n]);
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

Timer timer;

void callback(NetEntry entry){
    double t=timer.stop();
    print(Timer::format(t),"\t",entry.performance);
    timer.start();
}


int main(){
    srand(time(NULL));
    vec<size_t> netshape(5,5);

    Trainer trainer(netshape,NNet::Activation::relu,perform);
    trainer.gen_callback=callback;
    timer.start();
    trainer.train(100);
}

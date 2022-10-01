#include "imitator.hpp"

dvec randvec(size_t s){
    dvec v(s);
    for(int n=0;n<s;n++){
        v[n]=(double)rand()/RAND_MAX;
    }
    return v;
}

double ImitatorTrainer::perform(NNet& net){
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
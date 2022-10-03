#include "imitator.hpp"

//make a random vector of doubles of size 's'
dvec randvec(size_t s){
    dvec v(s);
    for(int n=0;n<s;n++){
        v[n]=(double)rand()/RAND_MAX;
    }
    return v;
}

double ImitatorTrainer::perform(const NNet& net){

    //prepare space in memory
    dvec* inputs=dvec::new_array(net.shape.layers[0],samples_per_net);
    dvec* target=dvec::new_array(net.shape.layers[net.shape.layers.size()-1],samples_per_net);
    dvec* output=dvec::new_array(net.shape.layers[net.shape.layers.size()-1],samples_per_net);

    double total_err=0;
    for(size_t n=0;n<samples_per_net;n++){

        inputs[n]=randvec(net.shape.layers[0]);
        target[n]=target_function(inputs[n]);
        output[n]=net.eval(inputs[n]);

        //represent error as distance squared b/w output and target
        dvec errv=output[n]-target[n];
        total_err+=dot(errv,errv);

    }
    dvec::delete_array(inputs);
    dvec::delete_array(target);
    dvec::delete_array(output);
    return total_err;
}

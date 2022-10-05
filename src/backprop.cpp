#include "backprop.hpp"

dvec mapvec(dvec v,double(*f)(double)){
    for(size_t n=0;n<v.size();n++){
        v[n]=f(v[n]);
    }
    return v;
}

NNet backprop(NNet& net,dvec in,dvec expected){
    size_t last_layer_idx=net.shape.layers.size()-2;
    //evaluate network on the input; however, unlike NNet::eval, keep track of the output of
    //each neuron
    bloc<dvec> neuron_outputs(net.shape.layers.size()-1);
    neuron_outputs[0]=mapvec(net.weights[0]*in+net.biases[0],
                             net.shape.actfunc.activate);
    for(size_t l=1;l<=last_layer_idx;l++){
        neuron_outputs[l]=mapvec(net.weights[l]*neuron_outputs[l-1]+net.biases[l],
                                 net.shape.actfunc.activate);
    }

    //calculate "delta" for each neuron. This is used to calculate the gradient later.
    //you can consider this to be d(error)/d(neuron)
    bloc<dvec> neuron_deltas(net.shape.layers.size()-1);
    neuron_deltas[last_layer_idx]=neuron_outputs[last_layer_idx]-expected;
    neuron_deltas[last_layer_idx]*=mapvec(neuron_outputs[last_layer_idx],
                                          net.shape.actfunc.derivative);
    for(int l=last_layer_idx-1;l>=0;l--){
        //NB: transpose(mat)*vec == vec*mat
        neuron_deltas[l]=neuron_deltas[l+1]*net.weights[l+1];
        neuron_deltas[l]*=mapvec(neuron_outputs[l],net.shape.actfunc.derivative);
    }

    //calculate the gradient from deltas. this is d(error)/d(weight)
    NNet ret(net.shape);
    for(int n=0;n<net.shape.layers[1];n++){
        for(int m=0;m<net.shape.layers[0];m++){
            ret.weights[0][n][m]=neuron_deltas[0][n]*in[m];
        }
    }
    for(int l=1;l<=last_layer_idx;l++){
        for(int n=0;n<net.shape.layers[l+1];n++){
            for(int m=0;m<net.shape.layers[l];m++){
                ret.weights[l][n][m]=neuron_deltas[l][n]*neuron_outputs[l-1][m];
            }
        }
    }
    swap(neuron_deltas,ret.biases);
    neuron_deltas.destroy();
    return ret;
}

void net_add(NNet* net_a,const NNet* net_b,double scalar){
    for(size_t l=0;l<net_a->shape.layers.size()-1;l++){
        net_a->weights[l]=net_a->weights[l]+net_b->weights[l]*scalar;
        net_a->biases[l]+=net_b->biases[l]*scalar;
    }
}

NNet* BackPropTrainer::train(){
    NNet* net=new NNet(net_shape);
    net->mutate(1.0);
    for(size_t g=0;g<gen_count;g++){
        Sample next=sample(g);
        NNet gradient=backprop(*net,next.first,next.second);
        for(size_t s=0;s<samples_per_gen-1;s++){
            next=sample(g);
            NNet grad2=backprop(*net,next.first,next.second);
            net_add(&gradient,&grad2,1.0/samples_per_gen);
        }
        net_add(net,&gradient,-learn_rate);
        gen_callback(this,g,net);
    }
    return net;
}

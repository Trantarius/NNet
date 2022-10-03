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
    vec<dvec> neuron_outputs(net.shape.layers.size()-1);
    neuron_outputs[0]=mapvec(net.weights[0]*in+net.biases[0],
                             net.shape.actfunc.activate);
    for(size_t l=1;l<=last_layer_idx;l++){
        neuron_outputs[l]=mapvec(net.weights[l]*neuron_outputs[l-1]+net.biases[l],
                                 net.shape.actfunc.activate);
    }

    dvec* neuron_deltas=new dvec[net.shape.layers.size()-1];
    neuron_deltas[last_layer_idx]=neuron_outputs[last_layer_idx]-expected;
    neuron_deltas[last_layer_idx]*=mapvec(neuron_outputs[last_layer_idx],
                                          net.shape.actfunc.derivative);
    for(int l=last_layer_idx-1;l>=0;l--){
        //NB: transpose(mat)*vec == vec*mat
        neuron_deltas[l]=neuron_deltas[l+1]*net.weights[l+1];
        neuron_deltas[l]*=mapvec(neuron_outputs[l],net.shape.actfunc.derivative);
    }

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
    std::swap(neuron_deltas,ret.biases);
    delete [] neuron_deltas;
    return ret;
}

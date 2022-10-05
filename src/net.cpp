 
#include "net.hpp"
#include <cmath>

//approximation of a normally distributed random number
//see https://en.wikipedia.org/wiki/Irwin-Hall_distribution#Approximating_a_Normal_distribution
double ih_rand(){
    double ret=0;
    for(int n=0;n<12;n++){
        ret+=(rand()/(double)RAND_MAX);
    }
    return ret-6;
}

NNet::NNet(Netshape shape):shape(shape){
    weights=bloc<dmat>(shape.layers.size()-1);
    biases=bloc<dvec>(shape.layers.size()-1);

    //start off with weights+biases that pass through some or all of the last layers output
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){
        weights[layer]=dmat::identity(shape.layers[layer+1],shape.layers[layer]);
        biases[layer]=dvec(shape.layers[layer+1]).fill(0);
    }
}
NNet::NNet(vec<size_t> shape,ActivationFunction actfunc):NNet(Netshape(shape,actfunc)){}

NNet::~NNet(){
    weights.destroy();
    biases.destroy();
}

void NNet::mutate(double sd){
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){

        for(size_t n=0;n<weights[layer].rows();n++){
            for(size_t m=0;m<weights[layer].cols();m++){
                (weights[layer])[n][m]+=ih_rand()*sd;
            }
        }

        for(size_t n=0;n<biases[layer].size();n++){
            (biases[layer])[n]+=ih_rand()*sd;
        }

    }
    //the network has changed, so the performance value is no longer valid
    performance=NAN;
}

dvec NNet::eval(dvec v) const {
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){
        //go from output of layer i to output of layer i+1
        v = weights[layer]*v + biases[layer];
        for(size_t n=0;n<v.size();n++){
            v[n]=shape.actfunc.activate(v[n]);
        }
    }
    return v;
}

NNet* NNet::clone() const {
    NNet* ret=new NNet(shape);
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){
        ret->weights[layer]=weights[layer];
        ret->biases[layer]=biases[layer];
    }
    return ret;
}

void NNet::copy(NNet& b){
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){
        weights[layer]=b.weights[layer];
        biases[layer]=b.biases[layer];
    }
}

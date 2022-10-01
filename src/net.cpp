 
#include "net.hpp"
#include <cmath>

double ih_rand(){
    double ret=0;
    for(int n=0;n<12;n++){
        ret+=(rand()/(double)RAND_MAX);
    }
    return ret-6;
}


NNet::NNet(vec<size_t> shape,double (*afunc)(double)):shape(shape),activation_function(afunc){
    weights=new dmat[shape.size()-1];
    biases=new dvec[shape.size()-1];

    for(size_t layer=0;layer<shape.size()-1;layer++){
        weights[layer]=dmat::identity(shape[layer+1],shape[layer]);
        biases[layer]=dvec(shape[layer+1]).fill(0);
    }
}

NNet::~NNet(){
    delete [] weights;
    delete [] biases;
}

void NNet::mutate(double sd){
    for(size_t layer=0;layer<shape.size()-1;layer++){

        for(size_t n=0;n<weights[layer].rows();n++){
            for(size_t m=0;m<weights[layer].cols();m++){
                (weights[layer])[n][m]+=ih_rand()*sd;
            }
        }

        for(size_t n=0;n<biases[layer].size();n++){
            (biases[layer])[n]+=ih_rand()*sd;
        }

    }
}

dvec NNet::eval(dvec v) const {
    for(size_t layer=0;layer<shape.size()-1;layer++){
        v = weights[layer]*v + biases[layer];
        for(size_t n=0;n<v.size();n++){
            v[n]=activation_function(v[n]);
        }
    }
    return v;
}

NNet* NNet::clone() const {
    NNet* ret=new NNet(shape,activation_function);
    for(size_t layer=0;layer<shape.size()-1;layer++){
        ret->weights[layer]=weights[layer];
        ret->biases[layer]=biases[layer];
    }
    return ret;
}

void NNet::copy(NNet& b){
    for(size_t layer=0;layer<shape.size()-1;layer++){
        weights[layer]=b.weights[layer];
        biases[layer]=b.biases[layer];
    }
}

#define E 2.71828182845904523536

double NNet::Activation::logistic(double x){
    return 1/(exp(-2*x)+1);
}

double NNet::Activation::sigmoid(double x){
    return (exp(2*x)-1)/(exp(2*x)+1);
}

double NNet::Activation::relu(double x){
    return x*(copysign(1,x)+1)/2;
}

double NNet::Activation::dying_sigmoid(double x){
    return x/(exp(x*x/(2*E)));
}

double NNet::Activation::softplus(double x){
    return log(exp(x)+1);
}

double NNet::Activation::swish(double x){
    return x/(exp(-x)+1);
}

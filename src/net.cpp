 
#include "net.hpp"


double ih_rand(){
    double ret=0;
    for(int n=0;n<12;n++){
        ret+=(rand()/(double)RAND_MAX);
    }
    return ret-6;
}


NNet::NNet(vec<size_t> shape,double (*afunc)(double)):shape(shape),activation_function(afunc){
    weights=new dmat*[shape.size()-1];
    biases=new dvec*[shape.size()-1];

    for(size_t layer=0;layer<shape.size()-1;layer++){
        weights[layer]=new dmat(shape[layer+1],shape[layer]);
        *weights[layer]=dmat::identity(shape[layer+1],shape[layer]);
        biases[layer]=new dvec(shape[layer+1]);
        biases[layer]->fill(0);
    }
}

NNet::~NNet(){
    for(size_t layer=0;layer<shape.size()-1;layer++){
        delete weights[layer];
        delete biases[layer];
    }
    delete [] weights;
    delete [] biases;
}

void NNet::mutate(double sd){
    for(size_t layer=0;layer<shape.size()-1;layer++){

        for(size_t n=0;n<weights[layer]->rows();n++){
            for(size_t m=0;m<weights[layer]->cols();m++){
                (*weights[layer])[n][m]+=ih_rand()*sd;
            }
        }

        for(size_t n=0;n<biases[layer]->size();n++){
            (*biases[layer])[n]+=ih_rand()*sd;
        }

    }
}

dvec NNet::eval(dvec v){
    for(size_t layer=0;layer<shape.size()-1;layer++){
        v = (*weights[layer])*v + (*biases[layer]);
        for(size_t n=0;n<v.size();n++){
            v[n]=activation_function(v[n]);
        }
    }
    return v;
}

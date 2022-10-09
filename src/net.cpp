 
#include "net.hpp"
#include <cmath>


uint64_t makeseed(){
    srand(time(NULL));
    return ((uint64_t)rand()<<32)^rand();
}

uint64_t xorshift64()
{
    static uint64_t state=makeseed();
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return state = x;
}

double randf(){
    return xorshift64()/(double)(~0ul);
    //return (double)rand()/RAND_MAX;
}

//approximation of a normally distributed random number
//see https://en.wikipedia.org/wiki/Irwin-Hall_distribution#Approximating_a_Normal_distribution
double ih_rand(){
    double ret=0;
    for(int n=0;n<12;n++){
        ret+=randf();
    }
    return ret-6;
}

//faster alternative to ih_rand
double randvar(){
    return 2*randf()-1;
}


NNet::NNet(Netshape shape):shape(shape){
    weights=bloc<dmat>(shape.layers.size()-1);
    biases=bloc<dvec>(shape.layers.size()-1);
    for(size_t layer=0;layer<shape.layers.size()-1;layer++){
        weights[layer]=dmat(shape.layers[layer+1],shape.layers[layer]);
        weights[layer].fill(0);
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

struct BinStream{
    void* ptr;
    template<typename T>
    void push(T arg){
        *(T*)ptr=arg;
        ptr=((T*)ptr+1);
    }
    template<typename T>
    T pop(){
        T ret=*(T*)ptr;
        ptr=((T*)ptr+1);
        return ret;
    }
};

//this must match what is found in namespace Activation in net.hpp
ActivationFunction id2afunc(uint8_t id){
    switch(id){
        case 1:return Activation::sigmoid;
        case 2:return Activation::tanh;
        case 3:return Activation::relu;
        case 4:return Activation::softplus;
        case 5:return Activation::swish;
        case 6:return Activation::wave;
        default:throw std::runtime_error("invalid activation function id: "+std::to_string(id));
    }
}

bloc<uchar> NNet::serialize(){
    size_t total_size=0;
    total_size+=sizeof(uint32_t);//number of layers;
    total_size+=sizeof(uint32_t)*shape.layers.size();//size of each layer
    total_size+=sizeof(uint8_t);//activation function (as enum)
    for(uint l=0;l<shape.layers.size()-1;l++){
        total_size+=sizeof(double)*shape.layers[l]*shape.layers[l+1];//weights
        total_size+=sizeof(double)*shape.layers[l+1];//biases
    };

    bloc<uchar> ret(total_size);
    BinStream stream;
    stream.ptr=ret.ptr;

    stream.push<uint32_t>(shape.layers.size());
    for(uint l=0;l<shape.layers.size();l++){
        stream.push<uint32_t>(shape.layers[l]);
    }
    stream.push<uint8_t>(shape.actfunc.id);

    for(uint l=0;l<shape.layers.size()-1;l++){
        void* wptr=&(weights[l]);
        size_t wsize=weights[l].rows()*weights[l].cols()*sizeof(double);
        memcpy(stream.ptr,wptr,wsize);
        stream.ptr=((uchar*)(stream.ptr)+wsize);

        void* bptr=biases[l].ptr();
        size_t bsize=biases[l].size()*sizeof(double);
        memcpy(stream.ptr,bptr,bsize);
        stream.ptr=((uchar*)(stream.ptr)+bsize);
    }

    return ret;
}

NNet NNet::deserialize(const bloc<uchar> serial){
    BinStream stream;
    stream.ptr=serial.ptr;

    size_t num_layers=stream.pop<uint32_t>();
    vec<size_t> layers(num_layers);
    for(uint l=0;l<num_layers;l++){
        layers[l]=stream.pop<uint32_t>();
    }
    ActivationFunction afunc=id2afunc(stream.pop<uint8_t>());

    NNet net(Netshape(layers,afunc));
    for(uint l=0;l<num_layers-1;l++){
        void* wptr=&(net.weights[l]);
        size_t wsize=net.weights[l].rows()*net.weights[l].cols()*sizeof(double);
        memcpy(wptr,stream.ptr,wsize);
        stream.ptr=((uchar*)(stream.ptr)+wsize);

        void* bptr=net.biases[l].ptr();
        size_t bsize=net.biases[l].size()*sizeof(double);
        memcpy(bptr,stream.ptr,bsize);
        stream.ptr=((uchar*)(stream.ptr)+bsize);
    }

    return net;
}

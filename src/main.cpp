#include <iostream>
#include "net.hpp"
#include "trainer.hpp"
#include <cmath>
#include <ctime>

double activation(double x){
    return sin(x);
}

dvec target_function(dvec in){
    return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
}

int main(){
    srand(time(NULL));
    vec<size_t> netshape(5,5);

    Trainer trainer(netshape,activation,target_function);
    trainer.train(999,netshape,activation);
}

#include <iostream>
#include "../CUtil/util.hpp"
#include "net.hpp"
#include "trainer.hpp"
#include <cmath>
#include <ctime>

double activation(double x){
    return sin(x);
}

int main(){
    srand(time(NULL));
    vec<size_t> netshape(5,5);

    Trainer trainer;
    trainer.train(999,netshape,activation);
}

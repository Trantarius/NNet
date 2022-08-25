#include <iostream>
#include "../CUtil/util.hpp"
#include "net.hpp"
#include <cmath>
#include <ctime>

double activation(double x){
    return sin(x);
}

int main(){
    srand(time(NULL));
    vec<size_t> netshape(5,4,3,4,5);

    NNet net(netshape,activation);
    net.mutate(1);

    dvec input(8,2,6,3,1);
    dvec output=net.eval(input);

    print(output);
}


#include "imitatordemo.hpp"
#include "convolverdemo.hpp"
#include "backprop.hpp"


int main(){
    //ImitatorDemo::demo();
    NNet net(Netshape(vec<size_t>(3,4,5,4,3,2),Activation::sigmoid));
    net.mutate(1.0);
    NNet gradient=backprop(net,dvec(0.1,0.2,0.3),dvec(0.5,0.4));
}


#include "imitatordemo.hpp"
#include "convolverdemo.hpp"
#include "backprop.hpp"
#include "pigtrap.hpp"


int main(){
    srand(time(NULL));
    NNet net(Netshape(vec<size_t>(8,7,6,7,8),Activation::relu));
    net.mutate(1.0);
    bloc<uchar> serial=net.serialize();
    writefile("out.nnet",serial);
    bloc<uchar> read=readfile("out.nnet");
    NNet deser=NNet::deserialize(read);

    dvec in(1,2,3,4,5,6,7,8);
    print(net.eval(in));
    print(deser.eval(in));
    pig_demo();
}

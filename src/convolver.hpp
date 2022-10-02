#pragma once
#include "net.hpp"
#include "montecarlo.hpp"
#include "../png++/png.hpp"

class ConvolverTrainer:public MonteCarloTrainer{
public:
    typedef png::image<png::rgb_pixel> Image;
    const size_t kernel_radius;
    size_t samples_per_net=10;
    const bloc<Image> inputs;
    const bloc<Image> outputs;
    virtual double perform(const NNet& net);

    Image eval_img(const NNet& net,const Image& input);

    ConvolverTrainer(Netshape shape,size_t kernel_radius,bloc<Image> inputs,bloc<Image> outputs);
};

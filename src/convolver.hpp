#pragma once
#include "net.hpp"
#include "trainer.hpp"
#include "../png++/png.hpp"

class ConvolverTrainer:public Trainer{
public:
    typedef png::image<png::rgb_pixel> Image;
    const size_t kernel_radius;
    const vec<Image> inputs;
    const vec<Image> outputs;
    virtual double perform(const NNet& net);

    Image eval_img(const NNet& net,const Image& input);

    ConvolverTrainer(size_t kernel_radius,vec<size_t> shape,activation_func_t act_func,vec<Image>&& inputs,vec<Image>&& outputs);
};

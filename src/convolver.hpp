#pragma once
#include "net.hpp"
#include "montecarlo.hpp"
#include "../png++/png.hpp"


/*
 * Trains a network as a convolutional filter (like gaussian blur or sobel filter).
 * The network is run on a square area around a target pixel and outputs a single pixel.
 * The network is run for each pixel in the image, creating an output image.
 * Limited to supervised training; ie there must be a predetermined list of expected
 * outputs to compare the network against.
 * Must be in 1 byte per channel RGB; network must have exactly 3 outputs, and the correct
 * number of inputs for the given kernel_radius (this is (kernel_radius*2+1)^2).
 */
class ConvolverTrainer:public MonteCarloTrainer{
public:
    typedef png::image<png::rgb_pixel> Image;
    //determines size of input area; ie 1=3x3, 2=5x5, 3=7x7, etc.
    const size_t kernel_radius;
    //number of randomly chosen images to test each network on
    size_t samples_per_net=10;
    const bloc<Image> inputs;
    //expected output; actual network output is compared to this to determine error
    const bloc<Image> outputs;
    virtual double perform(const NNet& net);

    //evaluate a network for every pixel in an image and return the result as an image
    Image eval_img(const NNet& net,const Image& input);

    ConvolverTrainer(Netshape shape,size_t kernel_radius,bloc<Image> inputs,bloc<Image> outputs);
};

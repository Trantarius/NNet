#pragma once
#include "util.hpp"
#include <cmath>

//bundles an activation function with its derivative, which can be used for backpropogation
//see namespace Activation below for examples
struct ActivationFunction{
    typedef double(*funcptr_t)(double);
    const funcptr_t activate;
    const funcptr_t derivative;
    ActivationFunction(const ActivationFunction& b):
        activate(b.activate),derivative(b.derivative){}
    ActivationFunction(funcptr_t act,funcptr_t der):
        activate(act),derivative(der){}
};

//determines the shape (ie, number of layers and size of each layer) and activation
//function of a network, for easy copyability
struct Netshape{
    //size of each layer. size of this vector determines the number of layers.
    vec<size_t> layers;
    ActivationFunction actfunc;

    Netshape(const Netshape& b):
        layers(b.layers),actfunc(b.actfunc){}

    Netshape(vec<size_t> layers,ActivationFunction actfunc):
        layers(layers),actfunc(actfunc){}
};

/*
 * A single instance of a neural network. This is a fairly standard feed-forward network.
 * Each 'neuron' creates its output as a weighted sum of the outputs of the previous layer; this
 * sum is then passed through an activation function. The first layer is a perceptron layer, ie
 * it just outputs the data input to the network (this layer's size is the size of the input
 * data, which is static). There are no 'neuron' objects, instead whole layers are collected as
 * a matrix of weights and a vector of biases; outputs are calculated all at once for the whole
 * layer.
 */
struct NNet{
    //last calculated performance metric for this network. will be NAN if and only if the
    //performance has not been calculated for this network.
    double performance=NAN;
    const Netshape shape;

    //arrays of weights and biases for the corresponding layers. since the first layer is
    //a perceptron layer, there are shape.layers.size()-1 matrices and vectors
    bloc<dmat> weights;
    bloc<dvec> biases;

    NNet(Netshape shape);
    NNet(vec<size_t> shape,ActivationFunction actfunc);
    ~NNet();

    //evaluates the network on a certain input (AKA, the whole point of the net)
    dvec eval(dvec in) const;
    //modifies every weight and bias by a normally distributed random amount, with given
    //standard deviation
    void mutate(double std_dev);
    NNet* clone() const;
    void copy(NNet& b);
};

//Some common activation functions; see https://en.wikipedia.org/wiki/Activation_function
namespace Activation{
    const ActivationFunction sigmoid=ActivationFunction(
        [](double x)->double{
            return 1/(exp(-x)+1);
        },
        [](double x)->double{
            double gx=1/(exp(-x)+1);
            return gx*(1-gx);
        }
    );
    const ActivationFunction tanh=ActivationFunction(
        [](double x)->double{
            return ::tanh(x);
        },
        [](double x)->double{
            double gx=::tanh(x);
            return 1-gx*gx;
        }
    );
    const ActivationFunction relu=ActivationFunction(
        [](double x)->double{
            //branchless?
            //return x*(copysign(1,x)+1)/2;
            return x>0?x:0;
        },
        [](double x)->double{
            //branchless?
            //return (copysign(1,x)+1)/2;
            return x>0?1:0;
        }
    );
    const ActivationFunction softplus=ActivationFunction(
        [](double x)->double{
            return log(1+exp(x));
        },
        [](double x)->double{
            return 1/(exp(-x)+1);
        }
    );
    const ActivationFunction swish=ActivationFunction(
        [](double x)->double{
            return x/(exp(-x)+1);
        },
        [](double x)->double{
            double ex=exp(-x);
            return (1+ex+x*ex)/((1+ex)*(1+ex));
        }
    );
    //I just made this one up
    const ActivationFunction wave=ActivationFunction(
        [](double x)->double{
            return x/exp(x*x/5.43656365692);
        },
        [](double x)->double{
            return (1-x*x/2.7182818284)/exp(x*x/5.43656365692);
        }
    );
}

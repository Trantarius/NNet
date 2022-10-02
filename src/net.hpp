#pragma once
#include "util.hpp"
#include <cmath>

struct ActivationFunction{
    typedef double(*funcptr_t)(double);
    const funcptr_t activate;
    const funcptr_t derivative;
    ActivationFunction(const ActivationFunction& b):
        activate(b.activate),derivative(b.derivative){}
    ActivationFunction(funcptr_t act,funcptr_t der):
        activate(act),derivative(der){}
};

struct Netshape{
    vec<size_t> layers;
    ActivationFunction actfunc;
    Netshape(const Netshape& b):layers(b.layers),actfunc(b.actfunc){}
    Netshape(vec<size_t> layers,ActivationFunction actfunc):
        layers(layers),actfunc(actfunc){}
};


struct NNet{
    const Netshape shape;
    dmat* weights;
    dvec* biases;

    NNet(Netshape shape);
    NNet(vec<size_t> shape,ActivationFunction actfunc);
    ~NNet();
    dvec eval(dvec in) const;
    void mutate(double std_dev);
    NNet* clone() const;
    void copy(NNet& b);
};

//Some common activation functions; see https://en.wikipedia.org/wiki/Activation_function
namespace Activation{
    const double E=2.71828182845904523536;
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
}

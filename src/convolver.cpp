#include "convolver.hpp"
#include <cmath>

ConvolverTrainer::ConvolverTrainer(Netshape shape,size_t kernel_radius,bloc<Image> inputs,
                                   bloc<Image> outputs):
    MonteCarloTrainer(shape),kernel_radius(kernel_radius),
    inputs(inputs),outputs(outputs){}


int clampi(int x,int low,int high){
    return x<low?low:(x>=high?high-1:x);
}

double clampf(double x,double low,double high){
    return x<low?low:(x>=high?nexttoward(high,low):x);
}

ConvolverTrainer::Image ConvolverTrainer::eval_img(const NNet& net,const Image& input){
    Image result(input.get_width(),input.get_height());
    dvec invec((kernel_radius*2+1)*(kernel_radius*2+1)*3);
    for(int x=0;x<result.get_width();x++){
        for(int y=0;y<result.get_height();y++){

            int idx=0;
            for(int dx=-kernel_radius;dx<=kernel_radius;dx++){
                for(int dy=-kernel_radius;dy<=kernel_radius;dy++){
                    int tx=clampi(x+dx,0,result.get_width());
                    int ty=clampi(y+dy,0,result.get_height());
                    invec[idx++]=input[ty][tx].red   /256.0;
                    invec[idx++]=input[ty][tx].green /256.0;
                    invec[idx++]=input[ty][tx].blue  /256.0;
                }
            }

            dvec resultpxl=net.eval(invec);
            result[y][x].red   = (int)(clampf(resultpxl[0],0,1)*256);
            result[y][x].green = (int)(clampf(resultpxl[1],0,1)*256);
            result[y][x].blue  = (int)(clampf(resultpxl[2],0,1)*256);
        }
    }
    return result;
}

double ConvolverTrainer::perform(const NNet& net){
    double totalerr=0;
    for(size_t s=0;s<samples_per_net;s++){
        size_t i=rand()%inputs.size;

        Image result=eval_img(net,inputs[i]);

        ulong imgerr=0;
        for(int x=0;x<result.get_width();x++){
            for(int y=0;y<result.get_height();y++){
                int d_r=result[y][x].red-outputs[i][y][x].red;
                int d_g=result[y][x].green-outputs[i][y][x].green;
                int d_b=result[y][x].blue-outputs[i][y][x].blue;
                imgerr+= d_r*d_r + d_g*d_g + d_b*d_b;
            }
        }
        totalerr+=(double)imgerr/(result.get_width()*result.get_height());
    }
    return totalerr/(double)samples_per_net;
}

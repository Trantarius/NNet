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

    //for each pixel
    for(int x=0;x<result.get_width();x++){
        for(int y=0;y<result.get_height();y++){

            //evaluate the result for this pixel.
            //network will operate on the range [0,1); image is in the range [0,256)

            //a temporary copy of the local area in a format that can be used as network input
            dvec invec((kernel_radius*2+1)*(kernel_radius*2+1)*3);
            int idx=0;
            //for every nearby pixel (ie within kernel)
            for(int dx=-kernel_radius;dx<=kernel_radius;dx++){
                for(int dy=-kernel_radius;dy<=kernel_radius;dy++){

                    //if pixel is out of bounds, replace with nearest pixel inside bounds
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
        //pick random image to sample
        size_t i=rand()%inputs.size;

        Image result=eval_img(net,inputs[i]);

        //find the average pixel error in this image (as distance squared in RGB-space)
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
    //return average of the average errors (prevents larger images from having a more significant effect)
    return totalerr/(double)samples_per_net;
}














ConvolverBackPropTrainer::ConvolverBackPropTrainer(Netshape shape,size_t kernel_radius,bloc<Image> inputs,
                                   bloc<Image> outputs):
                                   Trainer(shape),kernel_radius(kernel_radius),
                                   inputs(inputs),outputs(outputs){}




ConvolverBackPropTrainer::Image ConvolverBackPropTrainer::eval_img(const NNet& net,const Image& input){
    Image result(input.get_width(),input.get_height());

    //for each pixel
    for(int x=0;x<result.get_width();x++){
        for(int y=0;y<result.get_height();y++){

            //evaluate the result for this pixel.
            //network will operate on the range [0,1); image is in the range [0,256)

            //a temporary copy of the local area in a format that can be used as network input
            dvec invec((kernel_radius*2+1)*(kernel_radius*2+1)*3);
            int idx=0;
            //for every nearby pixel (ie within kernel)
            for(int dx=-kernel_radius;dx<=kernel_radius;dx++){
                for(int dy=-kernel_radius;dy<=kernel_radius;dy++){

                    //if pixel is out of bounds, replace with nearest pixel inside bounds
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


NNet* ConvolverBackPropTrainer::backprop_img(const NNet& net,const Image& input,const Image& expected){
    NNet* avg_grad=new NNet(net.shape);
    for(size_t l=0;l<net.shape.layers.size()-1;l++){
        avg_grad->biases[l].fill(0);
        avg_grad->weights[l].fill(0);
    }
    //for each pixel
    for(int x=0;x<input.get_width();x++){
        for(int y=0;y<input.get_height();y++){

            //evaluate the gradient for this pixel.
            //network will operate on the range [0,1); image is in the range [0,256)

            //a temporary copy of the local area in a format that can be used as network input
            dvec invec((kernel_radius*2+1)*(kernel_radius*2+1)*3);
            int idx=0;
            //for every nearby pixel (ie within kernel)
            for(int dx=-kernel_radius;dx<=kernel_radius;dx++){
                for(int dy=-kernel_radius;dy<=kernel_radius;dy++){

                    //if pixel is out of bounds, replace with nearest pixel inside bounds
                    int tx=clampi(x+dx,0,input.get_width());
                    int ty=clampi(y+dy,0,input.get_height());

                    invec[idx++]=input[ty][tx].red   /256.0;
                    invec[idx++]=input[ty][tx].green /256.0;
                    invec[idx++]=input[ty][tx].blue  /256.0;
                }
            }

            dvec outvec(3);
            outvec[0]=expected[y][x].red   /256.0;
            outvec[1]=expected[y][x].green /256.0;
            outvec[2]=expected[y][x].blue  /256.0;

            NNet* gradient=backprop(net,invec,outvec);
            net_add(avg_grad,gradient,1.0/(input.get_width()*input.get_height()));
            delete gradient;
        }
    }
    return avg_grad;
}

double ConvolverBackPropTrainer::perform(const NNet& net){
    double totalerr=0;
    for(size_t s=0;s<samples_per_net;s++){
        //pick random image to sample
        size_t i=rand()%inputs.size;

        Image result=eval_img(net,inputs[i]);

        //find the average pixel error in this image (as distance squared in RGB-space)
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
    //return average of the average errors (prevents larger images from having a more significant effect)
    return totalerr/(double)samples_per_net;
}

NNet* ConvolverBackPropTrainer::train(){
    NNet* net=new NNet(net_shape);
    net->mutate(1.0);
    for(size_t g=0;g<gen_count;g++){
        NNet* avg_grad=new NNet(net->shape);
        for(size_t l=0;l<net->shape.layers.size()-1;l++){
            avg_grad->weights[l].fill(0);
            avg_grad->biases[l].fill(0);
        }
        for(size_t s=0;s<samples_per_gen;s++){
            size_t imgidx=rand()%inputs.size;
            NNet* gradient=backprop_img(*net,inputs[imgidx],outputs[imgidx]);
            net_add(avg_grad,gradient,1.0/samples_per_gen);
            delete gradient;
            img_callback(this,s,net);
        }
        net_add(net,avg_grad,-learn_rate);
        gen_callback(this,g,net);
        delete avg_grad;
    }
    return net;
}

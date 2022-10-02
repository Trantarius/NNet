#include <iostream>
#include "net.hpp"
#include "imitator.hpp"
#include "convolver.hpp"
#include <cmath>
#include <ctime>
#include <filesystem>
#include <cstdlib>

typedef png::image<png::rgb_pixel> Image;
using std::filesystem::path;
using std::filesystem::directory_entry;
using std::filesystem::directory_iterator;

dvec target_function(dvec in){
    return dvec(sin(in[1]),cos(in[0]),sin(in[4]),cos(in[2]),sin(in[3]));
}

Timer timer;

void callback(NetEntry entry){
    double t=timer.stop();
    print(Timer::format(t),"\t",entry.performance);
    timer.start();
}


void print_loadbar(double completion){
    string out="\r[";
    int total_length=64;
    int filled=total_length*completion;
    for(int n=0;n<filled;n++){
        out+='#';
    }
    for(int n=0;n<total_length-filled;n++){
        out+=' ';
    }
    out+="]";
    std::cout<<out;
}


bloc<Image> load_images(path dir){
    size_t file_count=0;
    for(directory_entry entry:directory_iterator(dir)){
        if(entry.path().extension()!=string(".png")){
            continue;
        }
        file_count++;
    }
    bloc<Image> images(file_count);
    size_t idx=0;
    for(directory_entry entry:directory_iterator(dir)){
        try{
        if(entry.path().extension()!=string(".png")){
            continue;
        }
        //system(("magick "+entry.path().string()+" -strip "+entry.path().string()).c_str());
        images[idx++].read(entry.path());
        print_loadbar(idx/(double)file_count);
        }catch(png::error e){
            print("file: ",entry.path().string());
        }
    }
    return images;
}

int main(){
    srand(time(NULL));
    vec<size_t> netshape(9,7,5,3);

    bloc<Image> inputs=load_images("noiseimages");
    bloc<Image> outputs=load_images("images");

    ConvolverTrainer trainer(1,netshape,NNet::Activation::dying_sigmoid,inputs,outputs);
    trainer.gen_callback=callback;
    trainer.keep_ratio=10;

    timer.start();
    trainer.train(100);
}

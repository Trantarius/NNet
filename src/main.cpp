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

void gen_callback(Trainer* trainer,size_t n,NetEntry entry){
    double t=timer.stop();
    printw(16,Timer::format(t),entry.performance," "," "," "," ");
    timer.start();
}

void perf_callback(Trainer* trainer,size_t n,NetEntry entry){
    static std::mutex mtx;
    mtx.lock();
    print_loadbar((double)n/trainer->nets_per_gen);
    mtx.unlock();
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
    print("reading ",dir.string());
    for(directory_entry entry:directory_iterator(dir)){
        try{
            if(entry.path().extension()!=string(".png")){
                continue;
            }
            images[idx++].read(entry.path());
            print_loadbar(idx/(double)file_count);
        }catch(png::error e){
            print("\nerror: ",entry.path().string());
            exit(1);
        }
    }
    std::cout<<std::endl;
    return images;
}

int main(){
    srand(time(NULL));
    vec<size_t> netshape(27,3);

    bloc<Image> inputs=load_images("noiseimages");
    bloc<Image> outputs=load_images("images");

    ConvolverTrainer trainer(1,netshape,NNet::Activation::logistic,inputs,outputs);
    trainer.gen_callback=gen_callback;
    trainer.perf_callback=perf_callback;

    trainer.samples_per_net=10;
    trainer.nets_per_gen=1000;
    trainer.mutation_rate=0.1;
    trainer.keep_ratio=10;
    trainer.log_enabled=true;

    timer.start();
    trainer.train(100);
}

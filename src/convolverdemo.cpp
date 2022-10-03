#include "convolverdemo.hpp"

namespace ConvolverDemo{

    Timer timer;

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
        double t=timer.stop();
        printw(16,Timer::format(t),net->performance," "," "," "," ");
        timer.start();
    }

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
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

    void demo(){
        srand(time(NULL));
        Netshape netshape(vec<size_t>(27,3),Activation::tanh);

        bloc<Image> inputs=load_images("noiseimages");
        bloc<Image> outputs=load_images("images");

        ConvolverTrainer trainer(netshape,1,inputs,outputs);
        trainer.gen_callback=gen_callback;
        trainer.perf_callback=perf_callback;

        trainer.gen_count=100;
        trainer.samples_per_net=10;
        trainer.nets_per_gen=1000;
        trainer.mutation_rate=0.05;
        trainer.keep_ratio=10;
        trainer.log_enabled=true;

        timer.start();
        trainer.train();
    }
}


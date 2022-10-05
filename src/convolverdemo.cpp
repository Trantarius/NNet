#include "convolverdemo.hpp"

/*
 * Creates an image denoiser from a set of training images, and versions of those images with
 * noise already added.
 * Records progress in log.csv.
 * Doesn't really work, really just to compare to alternatives at this point
 */
namespace ConvolverDemo{

    Timer timer;

    //print the time it took to train the generation, and the best performance in that generation
    //also clears the loadbar
    void gen_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
        double t=timer.stop();
        printw(16,Timer::format(t),net->performance," "," "," "," ");
        timer.start();
    }

    //print a loadbar for how far along this generation we are
    void perf_callback(MonteCarloTrainer* trainer,size_t n,NNet* net){
        static std::mutex mtx;
        mtx.lock();
        print_loadbar((double)n/trainer->nets_per_gen);
        mtx.unlock();
    }

    //load a bunch of images. ignores anything that doesn't end in ".png".
    //very slow, prints a loadbar to show progress
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


    //print the time it took to train the generation, and the best performance in that generation
    //also clears the loadbar
    void gen_callback_bp(ConvolverBackPropTrainer* trainer,size_t n,NNet* net){
        double t=timer.stop();
        double perf=trainer->perform(*net);
        printw(16,Timer::format(t),perf," "," "," "," ");
        timer.start();
    }

    //print a loadbar for how far along this generation we are
    void img_callback_bp(ConvolverBackPropTrainer* trainer,size_t n,NNet* net){
        //static std::mutex mtx;
        //mtx.lock();
        print_loadbar((double)n/trainer->samples_per_gen);
        //mtx.unlock();
    }

    void demo_backprop(){
        srand(time(NULL));
        Netshape netshape(vec<size_t>(27,3),Activation::tanh);

        bloc<Image> inputs=load_images("noiseimages");
        bloc<Image> outputs=load_images("images");

        ConvolverBackPropTrainer trainer(netshape,1,inputs,outputs);
        trainer.gen_callback=gen_callback_bp;
        trainer.img_callback=img_callback_bp;

        trainer.gen_count=100;
        trainer.samples_per_net=100;
        trainer.samples_per_gen=100;
        trainer.learn_rate=0.05;

        timer.start();
        trainer.train();
    }
}


#include "../CUtil/util.hpp"
#include "../png++/png.hpp"
#include <filesystem>
#include <cmath>
#include <thread>
#include <list>

using namespace std::filesystem;
using Image=png::image<png::rgb_pixel>;
using Pixel=png::rgb_pixel;
const size_t thread_count=8;

double randf(){
    return (double)rand()/RAND_MAX;
}

double randg(){
    return (randf()+randf()+randf()+randf())-2;
}

int clamp(int x,int low,int high){
    return x<low?low:(x>high?high:x);
}

double clampf(double x,double low,double high){
    return x<low?low:(x>=high?nexttoward(high,low):x);
}


Pixel randc(Pixel base,double amount){
    double r=base.red;
    double g=base.green;
    double b=base.blue;

    r+=(randg()*amount/2)*r/255.0+(randg()*amount/2);
    g+=(randg()*amount/2)*g/255.0+(randg()*amount/2);
    b+=(randg()*amount/2)*b/255.0+(randg()*amount/2);

    base.red   = clamp(r,0,255);
    base.green = clamp(g,0,255);
    base.blue  = clamp(b,0,255);

    return base;
}


struct Task{
    double noise_amount;
    path infile;
    path outfile;

    Task(double noise_amount,path infile,path outfile):
        noise_amount(noise_amount),infile(infile),outfile(outfile){}

    void perform(){
        Image image(infile);
        for(int y=0;y<image.get_height();y++){
            for(int x=0;x<image.get_width();x++){

                image[y][x]=randc(image[y][x],noise_amount);
            }
        }
        image.write(outfile);
    }

    void perform_magick(){
        system(("magick "+infile.string()+" -attenuate "+tostr(noise_amount)+" +noise Gaussian "+outfile.string()).c_str());
    }

};

void do_all(std::list<Task>* tasks){
    while(!tasks->empty()){
        tasks->front().perform_magick();
        tasks->pop_front();
    }
}

int main(){
    path inpath("images");
    path outpath("noiseimages");

    if(!exists(outpath)){
        create_directory(outpath);
    }

    std::list<Task> tasks[thread_count];
    int next_list_idx=0;

    for(directory_entry entry:directory_iterator(inpath)){
        if(entry.path().extension()!=string(".png")){
            continue;
        }
        path outentry=outpath/entry.path().filename();
        tasks[next_list_idx].push_back(Task(randf(),entry.path(),outentry));
        next_list_idx=(next_list_idx+1)%thread_count;
    }

    std::thread threads[thread_count];

    for(int n=0;n<thread_count;n++){
        threads[n]=std::thread(do_all,&(tasks[n]));
    }
    for(int n=0;n<thread_count;n++){
        threads[n].join();
    }
}

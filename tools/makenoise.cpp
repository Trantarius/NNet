#include "../CUtil/util.hpp"
#include "../png++/png.hpp"
#include <filesystem>
#include <cmath>
#include <thread>
#include <list>

/*
 * image preprocessor.
 * converts jpg files to png, and saves another copy with noise added
 * designed to be used with images from https://cocodataset.org/#download
 */

using namespace std::filesystem;
using Image=png::image<png::rgb_pixel>;

const path orig_dir="val2017";
const path clean_dir="images";
const path noise_dir="noiseimages";
const size_t thread_count=16;

struct ImageTask:public Task{
    double noise_amount;
    path orig_file;
    path clean_file;
    path noise_file;

    ImageTask(double noise_amount,path orig_file,path clean_file,path noise_file):
        noise_amount(noise_amount),orig_file(orig_file),
        clean_file(clean_file),noise_file(noise_file){}

    void perform(){
        string convert_cmnd = "magick "+orig_file.string()+" "+clean_file.string();
        string noise_cmnd   = "magick "+orig_file.string()+" -attenuate "+tostr(noise_amount)+
                              " +noise Gaussian "+noise_file.string();
        system(convert_cmnd.c_str());
        system(noise_cmnd.c_str());

        Image img;
        img.read(clean_file);
        remove(clean_file);
        img.write(clean_file);

        img.read(noise_file);
        remove(noise_file);
        img.write(noise_file);
    }

};

int main(){

    freopen("/dev/null","w",stderr);

    create_directory(clean_dir);
    create_directory(noise_dir);

    Threadpool threadpool(thread_count);

    int count=0;
    for(directory_entry entry:directory_iterator(orig_dir)){
        if(entry.path().extension()!=string(".jpg")){continue;}

        threadpool.push(new ImageTask(
            rand()/(double)RAND_MAX,
            entry.path(),
            (clean_dir/entry.path().stem()).concat(".png"),
            (noise_dir/entry.path().stem()).concat(".png")
        ));
        count++;
    }

    while(!threadpool.idle()){
        print_loadbar((count-threadpool.tasks_left())/(double)count);
    }
    printw(66,"done");
}

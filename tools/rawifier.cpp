#include "../CUtil/util.hpp"
#include "../png++/png.hpp"
#include "../src/image.hpp"
#include <filesystem>
#include <cmath>
#include <thread>
#include <list>

using namespace std::filesystem;
using Png=png::image<png::rgb_pixel>;
const size_t thread_count=16;

struct Task{
    path infile;
    path outfile;

    Task(path infile,path outfile):infile(infile),outfile(outfile){}

    void perform(){
        Png pngimg(infile);
        bloc imgdata(2*sizeof(size_t)+sizeof(Pixel)*pngimg.get_width()*pngimg.get_height());
        Pixel* dloc=(Pixel*)(&imgdata+2*sizeof(size_t));
        Image image(dloc,pngimg.get_width(),pngimg.get_height());
        for(int y=0;y<pngimg.get_height();y++){
            for(int x=0;x<pngimg.get_width();x++){

                image[x][y]=Pixel(pngimg[y][x].red,pngimg[y][x].green,pngimg[y][x].blue);
            }
        }
        writefile(outfile,imgdata);
        image.data=nullptr;//to prevent it from handling the memory
        imgdata.destroy();
    }
};

void do_all(std::list<Task>* tasks){
    while(!tasks->empty()){
        tasks->front().perform();
        tasks->pop_front();
    }
}

int main(){
    path inpath("noiseimages");
    path outpath("rawimages");

    if(!exists(outpath)){
        create_directory(outpath);
    }

    std::list<Task> tasks[thread_count];
    int next_list_idx=0;

    for(directory_entry entry:directory_iterator(inpath)){
        if(entry.path().extension()!=string(".png")){
            continue;
        }
        path outentry=outpath/entry.path().stem();
        tasks[next_list_idx].push_back(Task(entry.path(),outentry));
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

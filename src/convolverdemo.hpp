#pragma once
#include "convolver.hpp"
#include <cmath>
#include <ctime>
#include <filesystem>
#include <cstdlib>

namespace ConvolverDemo{
    typedef png::image<png::rgb_pixel> Image;
    typedef std::filesystem::path path;
    typedef std::filesystem::directory_entry directory_entry;
    typedef std::filesystem::directory_iterator directory_iterator;

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NNet* entry);

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NNet* entry);


    bloc<Image> load_images(path dir);

    void demo();
}

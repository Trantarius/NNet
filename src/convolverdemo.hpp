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

    dvec target_function(dvec in);

    void gen_callback(MonteCarloTrainer* trainer,size_t n,NetEntry entry);

    void perf_callback(MonteCarloTrainer* trainer,size_t n,NetEntry entry);


    bloc<Image> load_images(path dir);

    void demo();
}

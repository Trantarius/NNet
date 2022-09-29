#include <utility>
#include <cstdlib>

struct Image{
    double* data=nullptr;
    struct{
        size_t x=0,y=0;
    }size;

    Image(){}
    Image(const Image& b)=delete;
    Image(Image&& b){
        *this=std::move(b);
    }
    Image(double* data,size_t sizex,size_t sizey):data(data){
        size.x=sizex;
        size.y=sizey;
    }
    Image(size_t sizex,size_t sizey):Image(new double[sizex*sizey],sizex,sizey){}
    ~Image(){
        if(data!=nullptr){
            delete [] data;
        }
    }

    void operator = (const Image& b) = delete;
    void operator = (Image&& b){
        data=b.data;
        size=b.size;
        b.data=nullptr;
        b.size.x=0;b.size.y=0;
    }

    double* operator[](size_t idx){
        return data+(idx*size.y);
    }
    const double* operator[](size_t idx) const {
        return data+(idx*size.y);
    }
};

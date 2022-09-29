#include <utility>
#include <cstdlib>

struct Pixel{
    u_char r=0,g=0,b=0;
    Pixel(){}
    Pixel(u_char r,u_char g,u_char b):r(r),g(g),b(b){}
};

struct Image{

    Pixel* data=nullptr;
    struct{
        size_t x=0,y=0;
    }size;

    Image(){}
    Image(const Image& b)=delete;
    Image(Image&& b){
        *this=std::move(b);
    }
    Image(Pixel* data,size_t sizex,size_t sizey):data(data){
        size.x=sizex;
        size.y=sizey;
    }
    Image(size_t sizex,size_t sizey):Image(new Pixel[sizex*sizey],sizex,sizey){}
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

    Pixel* operator[](size_t idx){
        return data+(idx*size.y);
    }
    const Pixel* operator[](size_t idx) const {
        return data+(idx*size.y);
    }
};

#pragma once
#include "net.hpp"
#include "montecarlo.hpp"


struct PigBoard{
    enum{PIG=-1,EMPTY=0,WALL=1};
    enum STATE{PIG_WIN,PIG_LOSE,PLAYING};

    const int size;
    dvec board_data;
    struct{int row,col;}pig;
    PigBoard(int size);
    double* operator[](int row){
        return &(board_data[row*size]);
    }

    STATE move_pig();

    static PigBoard random(int size);
};

string tostr(PigBoard& board);

class PigTrapTrainer:public MonteCarloTrainer{
public:
    //number of games to play to evaluate net performance
    size_t samples_per_net=100;
    const size_t gridsize=10;
    virtual double perform(const NNet& net);

    PigTrapTrainer(size_t gridsize,Netshape shape):
    MonteCarloTrainer(shape),gridsize(gridsize){
        sort_mode=SORT_MODE::DESCENDING;
    }
};

void pig_demo();

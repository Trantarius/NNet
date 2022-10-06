#include "pigtrap.hpp"

PigBoard::PigBoard(int size):size(size),board_data(size*size){
    board_data.fill(0);
}

string tostr(PigBoard& board){
    string ret;
    for(int r=0;r<board.size;r++){
        for(int c=0;c<board.size;c++){
            if(board[r][c]==PigBoard::EMPTY){
                ret+='.';
            }else if(board[r][c]==PigBoard::WALL){
                ret+='#';
            }else if(board[r][c]==PigBoard::PIG){
                ret+='O';
            }else{
                ret+='?';
            }
            ret+=' ';
        }
        ret+='\n';
    }
    return ret;
}

PigBoard::STATE PigBoard::move_pig(){
    auto in_bounds=[&](int r,int c){return !(r<0||r>=size||c<0||c>=size);};

    if(pig.row<=0||pig.row>=size-1||pig.col<=0||pig.col>=size-1){return PIG_WIN;}

    enum:uchar{UP=1,DOWN=1<<1,LEFT=1<<2,RIGHT=1<<3};

    mat<uchar> path_field(size);
    path_field.fill(0);

    if(operator[](pig.row-1)[pig.col]==0){
        path_field[pig.row-1][pig.col]=UP;
    }
    if(operator[](pig.row+1)[pig.col]==0){
        path_field[pig.row+1][pig.col]=DOWN;
    }
    if(operator[](pig.row)[pig.col-1]==0){
        path_field[pig.row][pig.col-1]=LEFT;
    }
    if(operator[](pig.row)[pig.col+1]==0){
        path_field[pig.row][pig.col+1]=RIGHT;
    }

    auto make_move=[&](uchar move){
        if(move==UP){
            operator[](pig.row)[pig.col]=0;
            pig.row-=1;
            operator[](pig.row)[pig.col]=-1;
        }else if(move==DOWN){
            operator[](pig.row)[pig.col]=0;
            pig.row+=1;
            operator[](pig.row)[pig.col]=-1;
        }else if(move==LEFT){
            operator[](pig.row)[pig.col]=0;
            pig.col-=1;
            operator[](pig.row)[pig.col]=-1;
        }else if(move==RIGHT){
            operator[](pig.row)[pig.col]=0;
            pig.col+=1;
            operator[](pig.row)[pig.col]=-1;
        }
    };

    mat<uchar> tmp_field=path_field;
    bool changed=true;
    while(changed){
        changed=false;
        for(int e=0;e<size-1;e++){
            if(path_field[0][e]){
                make_move(path_field[0][e]);
                return PLAYING;
            }
            if(path_field[size-1][e+1]){
                make_move(path_field[size-1][e+1]);
                return PLAYING;
            }
            if(path_field[e][size-1]){
                make_move(path_field[e][size-1]);
                return PLAYING;
            }
            if(path_field[e+1][0]){
                make_move(path_field[e+1][0]);
                return PLAYING;
            }
        }
        for(int r=0;r<size;r++){
            for(int c=0;c<size;c++){
                if(path_field[r][c]){continue;}
                if(operator[](r)[c]==1){continue;}
                if(in_bounds(r-1,c) && path_field[r-1][c]){
                    tmp_field[r][c]=path_field[r-1][c];
                    changed=true;
                }else if(in_bounds(r+1,c) && path_field[r+1][c]){
                    tmp_field[r][c]=path_field[r+1][c];
                    changed=true;
                }else if(in_bounds(r,c-1) && path_field[r][c-1]){
                    tmp_field[r][c]=path_field[r][c-1];
                    changed=true;
                }else if(in_bounds(r,c+1) && path_field[r][c+1]){
                    tmp_field[r][c]=path_field[r][c+1];
                    changed=true;
                }
            }
        }
        path_field=tmp_field;
    }
    return PIG_LOSE;
}

PigBoard PigBoard::random(int size){
    PigBoard board(size);
    constexpr auto randf=[](){return rand()/(double)RAND_MAX;};

    for(int r=0;r<size;r++){
        for(int c=0;c<size;c++){
            if(randf()>0.75){
                board[r][c]=1;
            }
        }
    }

    int pigx=(int)((randf()/2+0.25)*size);
    int pigy=(int)((randf()/2+0.25)*size);
    board[pigy][pigx]=-1;
    board.pig.row=pigy;
    board.pig.col=pigx;
    return board;
}


double PigTrapTrainer::perform(const NNet& net){

    auto apply_wall=[](dvec& board,dvec& output){
        for(int i=0;i<output.size();i++){
            int maxidx=0;
            double max=output[0];
            for(size_t n=1;n<output.size();n++){
                if(output[n]>max){
                    maxidx=n;
                    max=output[n];
                }
            }
            if(board[maxidx]==0){
                board[maxidx]=1;
                break;
            }else{
                output[maxidx]=-(1<<31);
            }
        }
    };

    double avg=0;
    for(int s=0;s<samples_per_net;s++){
        int score=0;
        PigBoard board=PigBoard::random(gridsize);
        PigBoard::STATE state=PigBoard::PLAYING;
        while(state==PigBoard::PLAYING){
            score++;
            dvec out=net.eval(board.board_data);
            apply_wall(board.board_data,out);
            state=board.move_pig();
        }
        if(state==PigBoard::PIG_LOSE){
            score+=gridsize;
        }
        avg+=score;
    }

    return avg/samples_per_net;
}

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

void pig_demo(){
    srand(time(NULL));
    Netshape netshape(vec<size_t>(64,64,48,32,48,64),Activation::tanh);

    PigTrapTrainer trainer(8,netshape);
    trainer.gen_callback=gen_callback;
    trainer.perf_callback=perf_callback;

    trainer.gen_count=100;
    trainer.samples_per_net=100;
    trainer.nets_per_gen=1000;
    trainer.mutation_rate=0.05;
    trainer.keep_ratio=10;
    trainer.log_enabled=true;

    timer.start();
    trainer.train();
}

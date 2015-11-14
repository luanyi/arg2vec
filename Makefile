CC=/g/ssli/sw/pkgs/gcc/4.9.3/bin/gcc
LIBS=-L../cnn/build/cnn -L/g/ssli/sw/pkgs/boost159/stage/lib -lcnn -lboost_filesystem -lboost_serialization -lboost_program_options -lstdc++ -lm
CFLAGS=-I../cnn/eigen -I../cnn -I/g/ssli/sw/pkgs/boost159 -I../cnn/external/easyloggingpp/src -std=gnu++11 -g
OBJ=util.o training.o arg2vec.o main_arg2vec.o

all: main-arg2vec

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $< 

main-arg2vec: main-arg2vec.o training.o util.o arg2vec.hpp
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.*~ main-arg2vec


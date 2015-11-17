#ifndef TRAINING_HPP
#define TRAINING_HPP

#include "arg2vec.hpp"
#include "util.hpp"
int train(char* ftrn, char* fdev, unsigned argsize, unsigned depsize, unsigned predsize, unsigned hidden, unsigned NCONTEXT, unsigned nneg, unsigned report_every_i, string flag, float lr0, bool use_adagrad, string logpath, string fmodel);
// int train(char* ftrn, char* fdev, unsigned nlayers = 2, 
// 	  unsigned inputdim = 16, unsigned hiddendim = 48, 
// 	  string flag = "output", float lr0 = 0.1, 
// 	  bool use_adagrad = false, string logpath="", string fmodel = "");

#endif

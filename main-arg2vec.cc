#include "util.hpp"
#include "training.hpp"
#include <stdlib.h>

// main function
int main(int argc, char** argv) {
  
  // initialize cnn
  cnn::Initialize(argc, argv);
  // check arguments
  cout << "Number of arguments " << argc << endl;
  if (argc < 5) {
    cerr << "============================\n"
	 << "Usage: \n" 
	 << "\t" << argv[0] 
	 << " train train_file dev_file flag \n\t\t[path] [input_dim] [hidden_dim] [learn_rate] [use_adagrad] [model_prefix]\n"
	 << "\t" << argv[0] 
	 << " test model_prefix test_file flag\n"
	 << "\t" << argv[0]
	 << " sample model_prefix test_file flag\n";
    return -1;
  }
  // parse command arguments
  string cmd = argv[1];
  if (cmd == "train"){
    cout << "Task: " << argv[1] <<endl;
    char* ftrn = argv[2];
    char* fdev = argv[3];
    string flag = string(argv[4]);
    unsigned argdim = 50;
    unsigned depdim = 30;
    unsigned hidden = 50;
    unsigned preddim = 80;
    unsigned NCONTEXT = 2;
    unsigned nneg = 5;
    unsigned report_every_i = 500;
    string path = "log";
    float lr0 = 0.1; // initial learning rate
    bool use_adagrad = true;
    string fmodel("");
    if (argc >= 6) path = string(argv[5]);
    if (argc >= 7) hidden = atoi(argv[6]);
    if (argc >= 8) report_every_i = atoi(argv[7]);
    if (argc >= 9) nneg = atoi(argv[8]);
    if (argc >= 10) argdim = atoi(argv[9]);
    if (argc >= 11) depdim = atoi(argv[10]);
    if (argc >= 12) preddim = atoi(argv[11]);
    if (argc >= 13) lr0 = atof(argv[12]);
    if (argc >= 14) use_adagrad = atoi(argv[13]);
    if (argc >= 15) fmodel = string(argv[14]);
    train(ftrn, fdev, argdim, depdim, preddim, hidden, NCONTEXT, nneg, report_every_i, flag, lr0, use_adagrad, path, fmodel);
  }
  else{
    cerr << "Unrecognized command " << argv[1]<<endl;
  }
}

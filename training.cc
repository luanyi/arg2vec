#include "training.hpp"

#include <boost/format.hpp>

// For logging
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

extern unsigned NCONV;

// ********************************************************
// train
// ********************************************************
int train(char* ftrn, char* fdev, unsigned argsize, unsigned depsize, unsigned predsize, unsigned hidden, unsigned NCONTEXT, unsigned nneg, unsigned report_every_i, string flag, float lr0, bool use_adagrad, string logpath, string fmodel){
  // initialize logging
  int argc = 1; 
  char** argv = new char* [1];
  START_EASYLOGGINGPP(argc, argv);
  delete[] argv;
  Dict Arg;
  Dict Dep;
  Dict Pred;
  Unigram unigram;
  string MODELPATH("models/" + logpath + '/');
  string LOGPATH("logs/" + logpath + '/');
  
  // ---------------------------------------------
  // predefined files
  ostringstream os;
  os << flag << '_' << argsize << '_' << depsize
     << '_' << predsize << '_' << hidden << '_' << nneg << '_' << lr0 << '_' << use_adagrad
     << "-pid" << getpid();
  const string fprefix = os.str();
  string fname = MODELPATH + fprefix;
  string flog = LOGPATH + fprefix + ".log";
  int nneg_dev = nneg*nneg;
  // int nneg_dev = nneg;
  cout << nneg_dev << endl;
  boost::filesystem::path dir(LOGPATH);
  if(!(boost::filesystem::exists(dir))){
    std::cout<< LOGPATH << " doesn't Exists"<<std::endl;

    if (boost::filesystem::create_directory(dir))
      std::cout << "....Successfully Created !" << "\n";
  }

  dir = MODELPATH;
  if(!(boost::filesystem::exists(dir))){
    std::cout<< MODELPATH << " doesn't Exists"<<std::endl;

    if (boost::filesystem::create_directory(dir))
      std::cout << "....Successfully Created !" << "\n";
  }

  // ----------------------------------------------
  // Pre-defined constants
  double best = 9e+99;
  unsigned dev_every_i_reports = 1;
  unsigned si = 0; // training.size();
  // --------------------------------------------
  // Logging
  el::Configurations defaultConf;
  // defaultConf.setToDefault();
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Format, 
  		  "%datetime{%h:%m:%s} %level %msg");
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Filename, flog.c_str());
  el::Loggers::reconfigureLogger("default", defaultConf);
  LOG(INFO) << "Training data: " << ftrn;
  LOG(INFO) << "Dev data: " << fdev;
  LOG(INFO) << "Parameters will be written to: " << fname;
  LOG(INFO) << "Negative sample number" << fname;
  // ---------------------------------------------
  // either create a dict or load one from the model file
  Corpus training, dev;
  if (fmodel.size() == 0){
    LOG(INFO) << "Create dict from training data ...";
    // read training data
    training = readData(ftrn, &Arg, &Dep, &Pred, &unigram, true);
    // no new word types allowed
    Arg.Freeze();
    Dep.Freeze();
    Pred.Freeze();
    unigram.Freeze();
    unigram.Normalize();
    // reading dev data
    dev = readData(fdev, &Arg, &Dep, &Pred, &unigram, false);
  } else {
    LOG(INFO) << "Load dict from pre-trained model: " << fmodel;
    ifstream in_arg(fmodel + "_arg.dict");
    ifstream in_dep(fmodel + "_dep.dict");
    ifstream in_pred(fmodel + "_pred.dict");
    ifstream in_unigram(fmodel + ".unigram");
    boost::archive::text_iarchive ia(in_arg);
    ia >> Arg; Arg.Freeze(); 
    boost::archive::text_iarchive id(in_dep);
    id >> Dep; Dep.Freeze(); 
    boost::archive::text_iarchive ipr(in_pred);
    ipr >> Pred; Pred.Freeze(); 
    boost::archive::text_iarchive iu(in_unigram);
    iu >> unigram; unigram.Freeze(); 
    training = readData(ftrn, &Arg, &Dep, &Pred, &unigram, false);
    dev = readData(fdev, &Arg, &Dep, &Pred, &unigram, false);
  }
  // get dict size
  unsigned argvsize = Arg.size();
  unsigned depvsize = Dep.size();
  unsigned predvsize = Pred.size();
  LOG(INFO) << "Arg size = " << argvsize;
  LOG(INFO) << "Dep size = " << depvsize;
  LOG(INFO) << "Pred size = " << predvsize;
  LOG(INFO) << "Hidden size = " << hidden;
  // save dict
  save_dict(fname + "_arg", Arg);
  save_dict(fname + "_dep", Dep);
  save_dict(fname + "_pred", Pred);
  save_unigram(fname, unigram);
  LOG(INFO) << "Save dict into: " << fname;
  LOG(INFO) << "Training set size: " << training.size();

  // ----------------------------------------------
  // define model
  Trainer* sgd = nullptr;
  cerr << "sgd" << endl;
  Model model;
  cerr << "model" << endl;
  Arg2vec lm(model, argvsize, depvsize, predvsize, argsize, depsize, predsize, hidden, NCONTEXT);
  // Load model
  if (fmodel.size() > 0){
    LOG(INFO) << "Load model from: " << fmodel;
      load_model(fprefix, model);
  } else {
    LOG(INFO) << "Randomly initializing model parameters ...";
  }
  
  sgd = new SimpleSGDTrainer(&model, 1e-6, lr0);
    
  // ---------------------------------------------
  // define the indices so we can shuffle the docs
  // vector<unsigned> order(training.size());
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true; int report = 0; unsigned lines = 0;
  // ---------------------------------------------
  // start training
  while(true) {
    Timer iteration("completed in");
    double loss = 0;
    // double lossd = 0;
    unsigned lines = 0, words = 0;
    //iterating over documents
    for (unsigned i = 0; i < report_every_i; ++i) { 
      //check if it's the number of documents
      if (si == training.size()) { 
        si = 0;
        if (first) { first = false; } 
	else { sgd->update_epoch(); }
        shuffle(order.begin(), order.end(), *rndeng);
	// cout<<"==SHUFFLE=="<<endl;
      }
      // get one document
      auto& conv = training[order[si]];
      // get negative samples
      unsigned wordl = 0;
      for (auto& sent : conv) wordl += (sent.size() - NCONTEXT);
      // get the right model
      // lm.BuildGraph(conv, unigram, nneg,cg);
      ComputationGraph cg;
      for (unsigned j=0; j<conv.size(); j++) {
	Sent sent = conv[j];
	lm.BuildGraphSent(sent, unigram, nneg, cg);
	// lm.BuildGraph(sent, unigram, cg);
	loss += as_scalar(cg.forward());
	cg.backward();
	sgd->update();
	cg.clear();
      }
      // for (unsigned l=0; l<conv.size(); l++) {
      // 	ComputationGraph cg;
      // 	Sent sent = conv[l];
      // 	lm.BuildGraphSent(sent, unigram, nneg, cg);
      // 	// lm.BuildGraph(sent, unigram, cg);
      // 	lossd += as_scalar(cg.forward());
      // }
      lines += 1;
      ++si;
      words += wordl;
    }
    cerr << words << endl;
    cerr << lines << endl;
    sgd->status();
    // cerr << lossd / words << endl;
    LOG(INFO) << " E = " 
	      << boost::format("%1.4f") % (loss / words) 
	      << " PPL = " 
	      << boost::format("%5.4f") % exp(loss / words) 
	      << ' ';
    
    // ----------------------------------------
    report++;
    int num = 30000;
    vector<double> p_vec;
    for (int i=0;i<num;i++) {p_vec.push_back(rand01());}

    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dwords = 0, docctr = 0;
      // dev.clear();
      // for (unsigned i=1; i<11; i++){
      // 	dev.push_back(training[training.size()-i])
      // }
      unsigned devl = dev.size();
      int j = 0;
      for (unsigned i=0; i<devl; i++) {
	// for each doc
	auto& conv = dev[i];
	for (unsigned m = 0; m < conv.size(); m++){
	  if (j > p_vec.size()-1){
	    j = 0;
	  }
	  ComputationGraph cg;
	  Sent sent = conv[m];
	  lm.BuildGraphSentp(sent, unigram, nneg_dev, cg, p_vec, j);
	  // lm.BuildGraph(sent, unigram, cg);
	  dloss += as_scalar(cg.forward());
	  j = j + sent.size()*nneg_dev;
	}
	for (auto& sent : conv) dwords += sent.size() - NCONTEXT;
      }
      cerr << dwords << endl;
      cerr << devl << endl;
      // print PPL on dev
      LOG(INFO) << "DEV[epoch=" 
		<< (lines / (double)training.size()) 
		<< "] E = "
		<< boost::format("%1.4f") % (dloss / dwords) 
		<< " PPL = " 
		<< boost::format("%5.4f") % exp(dloss / dwords) 
		<< " ("
		<< boost::format("%5.4f") % exp(best / dwords)
		<<") ";
      // Save model
      if (dloss < best) {
	best = dloss;
	LOG(INFO) << "Save model into: "<<fname;
	  save_model(fname, model);
      }
    }
  }
  delete sgd;
}

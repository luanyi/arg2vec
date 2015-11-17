#include "util.hpp"

unsigned NCONV = 2;

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model){
  ifstream in(fname + ".model");
  boost::archive::text_iarchive ia(in);
  ia >> model;
  return 0;
}

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model){
  ofstream out(fname + ".model");
  boost::archive::text_oarchive oa(out);
  oa << model; 
  out.close();
  return 0;
}

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d){
  fname += ".dict";
  ofstream out(fname);
  boost::archive::text_oarchive odict(out);
  odict << d; out.close();
  return 0;
}

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d){
  fname += ".dict";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> d; in.close();
  return 0;
}

// *******************************************************                     // save unigram from a archive file                                           // *******************************************************       
int save_unigram(string fname,  Unigram unigram){
  fname += ".unigram";
  ofstream out(fname);
  boost::archive::text_oarchive ounigram(out);
  ounigram << unigram; out.close();
  return 0;
}

// *******************************************************
// load unigram from a archive file
// *******************************************************
int load_unigram(string fname, Unigram unigram){
  fname += ".unigram";
  ifstream in(fname);
  boost::archive::text_iarchive iunigram(in);
  iunigram >> unigram; in.close();
  return 0;
}

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line, Dict* Arg, Dict* Pos, Dict* Dep, Dict* Pred, Unigram* unigram, bool update) {
  vector<string> strs;
  boost::split(strs, line, boost::is_any_of("\t"));
  vector<string > tokens;
  boost::split(tokens, strs[1], boost::is_any_of(" "));
  string arg, pos, dep, pred, token, arg_pos_dep;
  Sent res;
  Token Stoken, Etoken;
  Stoken.push_back(Arg->Convert("<s>"));
  Stoken.push_back(Pos->Convert("<s>"));
  Stoken.push_back(Dep->Convert("<s>"));
  Stoken.push_back(Pred->Convert("<s>"));
  Etoken.push_back(Arg->Convert("</s>"));
  Etoken.push_back(Pos->Convert("</s>"));
  Etoken.push_back(Dep->Convert("</s>"));
  Etoken.push_back(Pred->Convert("</s>"));

  res.push_back(Stoken);
  for (auto& token:tokens) {
    if (token.empty()) break;
    vector<string> elems;
    Token cToken;
    boost::split(elems, token, boost::is_any_of("|"));
    arg = elems[0];
    pos = elems[1];
    dep = elems[2];
    pred = elems[3];
    if (update){
      int argid = (Arg->Convert(arg));
      int posid = (Pos->Convert(pos));
      int depid = (Dep->Convert(dep));
      cToken.push_back(argid);
      cToken.push_back(posid);
      cToken.push_back(depid);
      cToken.push_back(Pred->Convert(pred));
      arg_pos_dep = to_string(argid) + '_' + to_string(posid) + '_' + to_string(depid);
      unigram->Convert(arg_pos_dep);
      res.push_back(cToken);
    } else {
      if (Arg->Contains(arg)){
	cToken.push_back(Arg->Convert(arg));
      }else{
	cToken.push_back(Arg->Convert("UNK"));
      }
      if (Pos->Contains(pos)){
        cToken.push_back(Pos->Convert(pos));
      }else{
        cToken.push_back(Pos->Convert("UNK"));
      }
      if (Dep->Contains(dep)){
        cToken.push_back(Dep->Convert(dep));
      }else{
        cToken.push_back(Dep->Convert("UNK"));
      }
      if (Pred->Contains(pred)){
        cToken.push_back(Pred->Convert(pred));
      }else{
        cToken.push_back(Pred->Convert("UNK"));
      }
      res.push_back(cToken);
    }
  }
  res.push_back(Etoken);
  return res;
}


// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, 
		Dict* arg,
		Dict* pos,
		Dict* dep,
		Dict* pred,
		Unigram* unigram,
		bool b_update){
  cerr << "reading data from "<< filename << endl;
  Corpus corpus;
  Doc doc;
  Sent sent;
  string line;
  int tlc = 0;
  int toks = 0;
  ifstream in(filename);
  while(getline(in, line)){
    ++tlc;
    if (line[0] != '='){
      sent = MyReadSentence(line, arg, pos, dep, pred, unigram, b_update);
      if (sent.size() > 0){
	doc.push_back(sent);
	toks += doc.back().size();
      } else {
	cerr << "Empty sentence: " << line << endl;
      }
    } else {
      if (doc.size() > 0){
	corpus.push_back(doc);
	doc.clear();
      } else {
	cerr << "Empty document " << endl;
      }
    }
  }
  if (doc.size() > 0){
    corpus.push_back(doc);
  }
  cerr << corpus.size() << " docs, " << tlc << " lines, " 
       << toks << " tokens, " << arg->size() << " arg types, " << pos->size() << " pos types," << dep->size() << " dependency types, " << pred->size() << " predicate types, " << unigram->size() << " arg-pos-dep types. " <<endl;
  return(corpus);
}

// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t){
  vector<float> vf;
  int dim = t.d.d[0];
  for (int idx = 0; idx < dim; idx++){
    vf.push_back(t.v[idx]);
  }
  return vf;
}

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path){
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    if (boost::filesystem::create_directory(dir)){
      std::cout << "....Successfully Created !" << "\n";
    }
  }
}

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus corpus, int thresh){
  Corpus newcorpus;
  for (auto& doc : corpus){
    if (doc.size() <= thresh){
      newcorpus.push_back(doc);
      continue;
    }
    Doc tmpdoc;
    int counter = 0;
    for (auto& sent : doc){
      if (counter < thresh){
	tmpdoc.push_back(sent);
	counter ++;
      } else {
	newcorpus.push_back(tmpdoc);
	tmpdoc.clear();
	tmpdoc.push_back(sent);
	counter = 0;
      }
    }
    if (tmpdoc.size() > 0){
      newcorpus.push_back(tmpdoc);
      tmpdoc.clear();
    }
  }
  return newcorpus;
}

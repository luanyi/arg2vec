#ifndef UTIL_HPP
#define UTIL_HPP

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/tensor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cnn;

// ********************************************************
// Predefined information, used for the entire project
// ********************************************************
// Redefined types
typedef vector<int> Token;
typedef vector<Token> Sent;
typedef vector<Sent> Doc;
typedef vector<Doc> Corpus;
typedef unordered_map<string, int> Map;

class Unigram {
public:
  Unigram() : frozen(false), map_unk(true), unk_id(-1), count(0) {}
  int count;
  void Freeze() { frozen = true; }

  inline int Convert(const std::string& word) {
    auto i = d_.find(word);
    if (i == d_.end()) {
      if (frozen) {
        if (map_unk) {
          return unk_id;
        }
        else {
	  std::cerr << "Unknown word encountered: " << word << std::endl;
          throw std::runtime_error("Unknown word encountered in frozen dictionary: " + word);
	}
      }
      words_.push_back(word);
      unigram_.push_back(1);
      count++;
      return d_[word] = words_.size() - 1;
    } else {
      int idx = (i->second);
      unigram_[idx]++;
      count++;
      return i->second;
    }
  }
  inline unsigned size() const { return words_.size(); }
  inline const string& Convert(const int& id) const {
    assert(id < (int)words_.size());
    return words_[id];
  }

  inline const vector<double>& GetUnigram() const {
    return unigram_;
  }

  inline const vector<string>& GetWord() const {
    return words_;
  }
  
  inline bool Contains(const std::string& words) {
    return !(d_.find(words) == d_.end());
  }

  void SetUnk(const std::string& word) {
    if (!frozen)
      throw std::runtime_error("Please call SetUnk() only after dictionary is frozen");
    unk_id = Convert(word);
    map_unk = true;
  }
  
  void Normalize() {
    for (auto &ele: unigram_) {
      ele /= count;
    }
  }
private:
  Map d_;
  vector<string> words_;
  vector<double> unigram_; 
  bool frozen;
  bool map_unk; // if true, map unknown word to unk_id                       
  int unk_id;

#if BOOST_VERSION >= 105600
  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & frozen;
    ar & map_unk;
    ar & unk_id;
    ar & words_;
    ar & d_;
    ar & unigram_;
    ar & count;
  }
#endif
};

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model);

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model);

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, cnn::Dict d);

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, cnn::Dict& d);

// *******************************************************
// save unigram from a archive file
// *******************************************************
int save_unigram(string fname, Unigram unigram);

// *******************************************************
// load unigram from a archive file
// *******************************************************
int load_unigram(string fname, Unigram unigram);

// *******************************************************
// read sentences and convect tokens to indices
// *******************************************************
Sent MyReadSentence(const std::string& line,
                    Dict* Arg, Dict* Pos, Dict* Dep, Dict* Pred,
                    bool update);

// *****************************************************
// 
// *****************************************************
Doc makeDoc();

// *****************************************************
// read training and dev data
// *****************************************************
Corpus readData(char* filename, Dict* arg, Dict* pos, Dict* dep, Dict* pred, Unigram* unigram, bool b_update = true);


// ******************************************************
// Convert 1-D tensor to vector<float>
// so we can create an expression for it
// ******************************************************
vector<float> convertT2V(const Tensor& t);

// ******************************************************
// Check the directory, if doesn't exist, create one
// ******************************************************
int check_dir(string path);

// ******************************************************
// Segment a long document into several short ones
// ******************************************************
Corpus segment_doc(Corpus doc, int thresh = 10);
#endif

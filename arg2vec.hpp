#ifndef DCLM_HIDDEN_HPP
#define DCLM_HIDDEN_HPP

#include "util.hpp"

class Arg2vec{
private:

  vector<LookupParameters*> p_arg; // argument embedding
  vector<LookupParameters*> p_dep; // dependency embedding
  LookupParameters* p_pred; // predicate embedding
  Parameters* p_wc; // context matrix: Arg*K1
  Parameters* p_wa; // context matrix: Arg*K1
  unsigned NCONTEXT;

public:
  Arg2vec();
  Arg2vec(Model& model, unsigned argvsize, unsigned depvsize, unsigned predvsize, unsigned argsize, unsigned depsize, unsigned predsize, unsigned hidden, unsigned ncontext){
    NCONTEXT = ncontext;
    for (int idx=0; idx<2;idx++){
      p_arg.push_back(model.add_lookup_parameters(argvsize, {argsize}));
      p_dep.push_back(model.add_lookup_parameters(depvsize, {depsize}));
    }
    p_pred = model.add_lookup_parameters(predvsize, {predsize});
    p_wc = model.add_parameters({predsize+2*(argsize+depsize), {argsize+depsize}});
    p_wa = model.add_parameters({argsize+depsize, hidden});
  } // END of a constructor
  
  Expression BuildGraph(const Doc& conv, Unigram& unigram, int nneg,  ComputationGraph& cg){ 
    vector<Expression> errs, i_vector(2), i_y_vector;
    Expression i_pred, v_c, v_0, v_0_neg, i_x_vector;
    Expression i_wc = parameter(cg, p_wc);
    Expression i_wa = parameter(cg, p_wa);
    unsigned words = 0;
    unsigned curp = NCONTEXT/2;
    vector<double> token_freq = unigram.GetUnigram();
    for (unsigned m = 0; m < conv.size(); m++){
      // for each sentence in this doc
      Sent sent = conv[m];
      unsigned slen = sent.size();
      for (unsigned k = 0; k < slen - NCONTEXT; k++){
	i_y_vector.clear();
	for (unsigned n = 0; n < NCONTEXT + 1; n++){
	  if (n != curp){ 
	    //CONTEXT ARGUMENT
	    i_vector[0] = lookup(cg, p_arg[0], sent[k+n][0]);
	    i_vector[1] = lookup(cg, p_dep[0], sent[k+n][1]);
	    Expression temp = concatenate(i_vector);
	    i_y_vector.push_back(temp);
	    // i_y_vector.push_back(concatenate(i_vector));
	  }
	  else{
	    //CURRENT ARGUMENT
	    i_vector[0] = lookup(cg, p_arg[1], sent[k+n][0]);
	    i_vector[1] = lookup(cg, p_dep[1], sent[k+n][1]);
	    i_x_vector = concatenate(i_vector);
	    i_pred = lookup(cg, p_pred, sent[k+n][2]);
	  }
	}
	i_y_vector.push_back(i_pred);
	Expression vec = concatenate(i_y_vector);
	v_c = transpose(vec) * i_wc;
	v_0 = transpose(i_x_vector) * i_wa;
	//NEGATIVE SAMPLES
	vector<Expression> neg_vec;
	for (unsigned i = 0; i < nneg; i++){
	  //NEGATIVE CONTEXT ARGUMENT
	  double p = rand01();
	  unsigned w = 0;
	  for (; w < token_freq.size(); w++){
	    p -= token_freq[w];
	    if (p < 0.0) break;
	  }
	  string token = unigram.Convert(w);
	  vector<string> elems;
	  boost::split(elems, token, boost::is_any_of("_"));
	  int argid = stoi(elems[0]);
	  int depid = stoi(elems[1]);
	  if (argid == sent[k+curp][0] && depid == sent[k+curp][2]) {
	    i--;
	    continue;
	  }
	  i_vector[0] = lookup(cg, p_arg[1], argid);
	  i_vector[1] = lookup(cg, p_dep[1], depid);
	  i_x_vector = concatenate(i_vector);
	  // v_0_neg = transpose(i_x_vector) * i_wa;
	  // neg_vec.push_back(logistic(-v_0_neg*transpose(v_c)));
	  neg_vec.push_back(log(logistic(-i_x_vector*transpose(v_c))));
	}
	Expression i_err = -log(logistic(v_c*transpose(v_0))) - (sum(neg_vec)/neg_vec.size());
	errs.push_back(i_err);
      }
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  } // END of BuildGraph

  Expression BuildGraphSent(const Sent& sent, Unigram& unigram, int nneg,  ComputationGraph& cg){ 
    vector<Expression> errs, i_vector(2), i_y_vector;
    Expression i_pred, v_c, v_0, v_0_neg, i_x_vector;
    Expression i_wc = parameter(cg, p_wc);
    Expression i_wa = parameter(cg, p_wa);
    unsigned words = 0;
    unsigned curp = NCONTEXT/2;
    vector<double> token_freq = unigram.GetUnigram();
    unsigned slen = sent.size();
    string cur_arg;
    for (unsigned k = 0; k < slen - NCONTEXT; k++){
      i_y_vector.clear();
      for (unsigned n = 0; n < NCONTEXT + 1; n++){
	if (n != curp){ 
	  //CONTEXT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[0], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[0], sent[k+n][1]);
	  Expression temp = concatenate(i_vector);
	  i_y_vector.push_back(temp);
	  // i_y_vector.push_back(concatenate(i_vector));
	}
	else{
	  //CURRENT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[1], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[1], sent[k+n][1]);
	  i_x_vector = concatenate(i_vector);
	  i_pred = lookup(cg, p_pred, sent[k+n][2]);
	}
      }
      i_y_vector.push_back(i_pred);
      Expression vec = concatenate(i_y_vector);
      v_c = transpose(vec) * i_wc;
      // v_0 = transpose(i_x_vector) * i_wa;
      v_0 = i_x_vector;
      //NEGATIVE SAMPLES
      vector<Expression> neg_vec;
      for (unsigned i = 0; i < nneg; i++){
	//NEGATIVE CONTEXT ARGUMENT
	unsigned w = 0;
	double p = rand01();
	for (; w < token_freq.size(); w++){
	  p -= token_freq[w];
	  if (p < 0.0) break;
	}
	if (w == token_freq.size()) w = w-1;
	string token = unigram.Convert(w);
	vector<string> elems;
	boost::split(elems, token, boost::is_any_of("_"));
	int argid = stoi(elems[0]);
	int depid = stoi(elems[1]);
	if (argid == sent[k+curp][0] && depid == sent[k+curp][2]) {
	  i--;
	  continue;
	}
	i_vector[0] = lookup(cg, p_arg[1], argid);
	i_vector[1] = lookup(cg, p_dep[1], depid);
	i_x_vector = concatenate(i_vector);
	// v_0_neg = transpose(i_x_vector) * i_wa;
	// neg_vec.push_back(log(logistic(-v_0_neg*transpose(v_c))));
	// neg_vec.push_back(log(logistic(-transpose(i_x_vector)*transpose(v_c))));
	neg_vec.push_back(log(logistic(-v_c*i_x_vector)));
      }
      // Expression i_err = -log(logistic(v_c*transpose(v_0))) - (sum(neg_vec)/neg_vec.size());
      Expression i_err = -log(logistic(v_c*v_0)) - sum(neg_vec);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  } // END of BuildGraph


  Expression BuildGraphSentp(const Sent& sent, Unigram& unigram, int nneg,  ComputationGraph& cg, vector<double>& p_vec, int j){ 
    vector<Expression> errs, i_vector(2), i_y_vector;
    Expression i_pred, v_c, v_0, v_0_neg, i_x_vector;
    Expression i_wc = parameter(cg, p_wc);
    Expression i_wa = parameter(cg, p_wa);
    unsigned words = 0;
    unsigned curp = NCONTEXT/2;
    vector<double> token_freq = unigram.GetUnigram();
    unsigned slen = sent.size();
    string cur_arg;
    for (unsigned k = 0; k < slen - NCONTEXT; k++){
      i_y_vector.clear();
      for (unsigned n = 0; n < NCONTEXT + 1; n++){
	if (n != curp){ 
	  //CONTEXT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[0], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[0], sent[k+n][1]);
	  Expression temp = concatenate(i_vector);
	  i_y_vector.push_back(temp);
	  // i_y_vector.push_back(concatenate(i_vector));
	}
	else{
	  //CURRENT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[1], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[1], sent[k+n][1]);
	  i_x_vector = concatenate(i_vector);
	  i_pred = lookup(cg, p_pred, sent[k+n][2]);
	}
      }
      i_y_vector.push_back(i_pred);
      Expression vec = concatenate(i_y_vector);
      v_c = transpose(vec) * i_wc;
      // v_0 = transpose(i_x_vector) * i_wa;
      v_0 = i_x_vector;
      //NEGATIVE SAMPLES
      vector<Expression> neg_vec;
      for (unsigned i = 0; i < nneg; i++){
	//NEGATIVE CONTEXT ARGUMENT
	double p = p_vec[j+i];
	unsigned w = 0;
	for (; w < token_freq.size(); w++){
	  p -= token_freq[w];
	  if (p < 0.0) break;
	}
	if (w == token_freq.size()) w = w-1;
	string token = unigram.Convert(w);
	vector<string> elems;
	boost::split(elems, token, boost::is_any_of("_"));
	int argid = stoi(elems[0]);
	int depid = stoi(elems[1]);
	if (argid == sent[k+curp][0] && depid == sent[k+curp][2]) {
	  i--;
	  j++;
	  continue;
	}
	i_vector[0] = lookup(cg, p_arg[1], argid);
	i_vector[1] = lookup(cg, p_dep[1], depid);
	i_x_vector = concatenate(i_vector);
	// v_0_neg = transpose(i_x_vector) * i_wa;
	// neg_vec.push_back(log(logistic(-v_0_neg*transpose(v_c))));
	// neg_vec.push_back(log(logistic(-transpose(i_x_vector)*transpose(v_c))));
	neg_vec.push_back(log(logistic(-v_c*i_x_vector)));
      }
      // Expression i_err = -log(logistic(v_c*transpose(v_0))) - (sum(neg_vec)/neg_vec.size());
      Expression i_err = -log(logistic(v_c*v_0)) - sum(neg_vec);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  } // END of BuildGraph


  Expression BuildGraph(const Sent& sent, Unigram& unigram, ComputationGraph& cg){ 
    vector<Expression> errs, i_vector(2), i_y_vector, softmax_vec;
    Expression i_pred, v_c, v_0, v_0_neg, i_x_vector;
    Expression i_wc = parameter(cg, p_wc);
    Expression i_wa = parameter(cg, p_wa);
    unsigned words = 0;
    unsigned curp = NCONTEXT/2;
    vector<double> token_freq = unigram.GetUnigram();
    // for (unsigned m = 0; m < conv.size(); m++){
    //   cout << m << endl;
    unsigned slen = sent.size();
    // cout << m << endl;
    for (unsigned k = 0; k < slen - NCONTEXT; k++){
      i_y_vector.clear();
      for (unsigned n = 0; n < NCONTEXT + 1; n++){
	if (n != curp){ 
	  //CONTEXT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[0], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[0], sent[k+n][1]);
	  i_y_vector.push_back(concatenate(i_vector));
	}
	else{
	  //CURRENT ARGUMENT
	  i_vector[0] = lookup(cg, p_arg[1], sent[k+n][0]);
	  i_vector[1] = lookup(cg, p_dep[1], sent[k+n][1]);
	  i_x_vector = concatenate(i_vector);
	  i_pred = lookup(cg, p_pred, sent[k+n][2]);
	}
      }
      i_y_vector.push_back(i_pred);
      Expression vec = concatenate(i_y_vector);
      v_c = transpose(vec) * i_wc;
      // v_0 = transpose(i_x_vector) * i_wa;
      softmax_vec.clear();
      // softmax_vec.push_back(v_c*transpose(v_0));
      softmax_vec.push_back(v_c*i_x_vector);
      for (unsigned i = 0; i < token_freq.size(); i++){
	string token = unigram.Convert(i);
	vector<string> elems;
	boost::split(elems, token, boost::is_any_of("_"));
	int argid = stoi(elems[0]);
	int depid = stoi(elems[1]);
	if (argid == sent[k+curp][0] && depid == sent[k+curp][2]) {
	  continue;
	}
	i_vector[0] = lookup(cg, p_arg[1], argid);
	i_vector[1] = lookup(cg, p_dep[1], depid);
	i_x_vector = concatenate(i_vector);
	// v_0_neg = transpose(i_x_vector) * i_wa;
	// softmax_vec.push_back(v_c*transpose(v_0_neg));
	softmax_vec.push_back(v_c*i_x_vector);
      }
      Expression obj_vec = concatenate(softmax_vec);
      Expression i_err = pickneglogsoftmax(obj_vec, 0);
      errs.push_back(i_err);
    }
    Expression i_nerr = sum(errs);
    return i_nerr;
  } // END of BuildGraph
};

#endif

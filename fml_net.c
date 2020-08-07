/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_net.h * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_internal.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

fml_net* fml_net_create(int n, ...){
  va_list vargs;
  int i;
  fml_net* ret = malloc((sizeof *ret) + (sizeof *ret->layers)*n);
  fml_layer_header** curr = ret->layers;

  va_start(vargs, n);
  for(i = 0; i < n; ++curr, ++i){
    *curr = va_arg(vargs, fml_layer_header*);
  }

  va_end(vargs);

  ret->n_layers = n;
  return ret;
}

void fml_net_set_learning_rate(fml_net* net, double learning_rate){
  net->learning_rate = learning_rate;
}

double fml_net_get_learning_rate(fml_net* net){
  return net->learning_rate;
}

void fml_net_set_cost_type(fml_net* net, fml_net_cost_type t){
  net->cost_type = t;
}

void fml_net_set_regularization_type(fml_net* net, fml_net_regularization_type t, double weight){
  net->regularization_type = t;
  net->regularization_weight = weight;
}

static void fml_net_learn(fml_net* net){
  return;
}

//void fml_layer_forward(fml_layer_header* layer_header, fml_layer_header* prev_layer_header);
static void fml_net_forward(fml_net* net, fml_data* data){
  unsigned int layer;
  fml_layer_header** curr = net->layers, **prev;

  fml_first_layer_forward(*curr, data);
  prev = curr;
  ++curr;
  for(layer = 2; layer < net->n_layers; ++layer, prev = curr, ++curr){
    fml_layer_forward(*curr, *prev);
  }

  return;
}

//double fml_layer_output_backprop(fml_layer_header* layer_header, fml_data* label, fml_net_cost_type cost_type);
//void fml_layer_backprop(fml_layer_header* layer, fml_layer* next_layer, fml_layer_header* prev_layer);
static double fml_net_backprop(fml_net* net, fml_data* sample, fml_data* label){
  double loss;
  fml_layer_header** curr;
  int layer = net->n_layers - 1;
  curr = net->layers + layer;

  loss = fml_layer_output_backprop(*curr, label, net->cost_type);
  for(--curr, --layer; layer > 0; --layer, --curr){
    fml_layer_backprop(*curr, *(curr + 1), *(curr - 1));
  }
  fml_layer_input_backprop(*curr, *(curr + 1), sample);

  return loss; 
}

void fml_train(fml_net* net, fml_dataset* dataset, unsigned int epochs, unsigned int minibatch_size){
  unsigned int epoch, iters_since_learned, i;
  fml_data* curr_sample, *curr_label;

  for(iters_since_learned = epoch = 0; epoch < epochs; ++epoch, ++iters_since_learned){
    for(curr_sample = dataset->samples, curr_label = dataset->labels, i = 0; i < dataset->n_train; ++i, ++curr_sample, ++curr_label){
      if(iters_since_learned == minibatch_size){
        fml_net_learn(net);
        iters_since_learned = 0;
      }
    
      fml_net_forward(net, curr_sample);
      fml_net_backprop(net, curr_sample, curr_label);
    }
  }
}

#if 0
void     fml_net_destroy(fml_net* net);
void     fml_net_dump(fml_net* net, FILE* file);
fml_net* fml_net_parse(FILE* file);
#endif


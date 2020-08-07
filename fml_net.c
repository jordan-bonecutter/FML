/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_net.h * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_internal.h"
#include <stdarg.h>
#include <stdlib.h>

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

static void fml_net_learn(fml_net* net){
  return;
}

//void fml_layer_forward(fml_layer_header* layer_header, fml_data* prev_layer_activations);
static void fml_net_forward(fml_net* net, fml_data* data){
  unsigned int layer;
  fml_layer_header** curr = net->layers, *prev;

  fml_layer_forward(curr, data);
  prev = curr;
  ++curr;
  for(layer = 2; layer < net->n_layers; ++layer, ++curr){
    fml_layer_forward(*curr, prev->activations);
    prev = curr;
  }

  return;
}

static void fml_net_backprop(fml_net* net, fml_data* data){
  return; 
}

void fml_train(fml_net* net, fml_dataset* dataset, unsigned int epochs, unsigned int minibatch_size){
  unsigned int epoch, iters_since_learned, layer, i;
  fml_data* curr_data;

  for(iters_since_learned = epoch = 0; epoch < epochs; ++epoch, ++iters_since_learned){
    for(curr_data = dataset->samples, i = 0; i < dataset->n_train; ++i, ++curr_data){
      if(iters_since_learned == minibatch_size){
        fml_net_learn(net);
        iters_since_learned = 0;
      }
    
      fml_net_forward(net, curr_data);
      fml_net_backprop(net, curr_data);
    }
  }
}

#if 0
void     fml_net_destroy(fml_net* net);
void     fml_net_dump(fml_net* net, FILE* file);
fml_net* fml_net_parse(FILE* file);
#endif


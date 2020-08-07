/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_internal.h  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_INTERNAL_INCLUDE_GUARDS
#define FML_INTERNAL_INCLUDE_GUARDS

#include <stdbool.h>

typedef struct{
  unsigned int n_dimensions;
  unsigned int* dimension;
}fml_data_shape;

typedef struct{
  fml_data_shape* shape;
  double* data;
}fml_data;

typedef struct{
  unsigned int n_data, n_train, n_validate, n_test;
  fml_data* samples;
  fml_data* labels;
}fml_dataset;

typedef enum{
  FML_NET_COST_TYPE_QUADRATIC,
  FML_NET_COST_TYPE_COUNT
}fml_net_cost_type;

typedef enum{
  FML_NET_REGULARIZATION_TYPE_NONE,
  FML_NET_REGULARIZATION_TYPE_L1,
  FML_NET_REGULARIZATION_TYPE_L2,
  FML_NET_REGULARIZATION_TYPE_COUNT
}fml_net_regularization_type;

typedef enum {
  FML_LAYER_TYPE_FULLY_CONNECTED,
  FML_LAYER_TYPE_SIGMOID,
  FML_LAYER_TYPE_TANH,
  FML_LAYER_TYPE_RELU,
  FML_LAYER_TYPE_LEAKY_RELU,
  FML_LAYER_TYPE_BATCH_NORMALIZE,
  FML_LAYER_TYPE_CONVOLUTION,
  FML_LAYER_TYPE_COUNT
} fml_layer_type;/* neural network layer type enum */

typedef struct{
  fml_layer_type layer_type;
  fml_data_shape* input_shape;
  fml_data* activations;
  fml_data* gradient;
}fml_layer_header;

void   fml_first_layer_forward(fml_layer_header* layer, fml_data* input);
void   fml_layer_forward(fml_layer_header* layer, fml_layer_header* prev_layer);
void   fml_layer_backprop(fml_layer_header* layer, fml_layer_header* next_layer, fml_layer_header* prev_layer);
double fml_layer_output_backprop(fml_layer_header* layer, fml_data* label, fml_net_cost_type cost_type);
void   fml_layer_input_backprop(fml_layer_header* layer, fml_layer_header* next_layer, fml_data* input);
void   fml_layer_learn(fml_layer_header* layer);

bool fml_data_have_same_shape(fml_data* d1, fml_data* d2);
unsigned int fml_data_size(fml_data* data);

typedef struct{
  fml_net_cost_type cost_type;
  fml_net_regularization_type regularization_type;
  unsigned int n_layers;
  double learning_rate, regularization_weight;
  fml_layer_header** layers;
}fml_net;

#endif


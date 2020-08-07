/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_internal.h  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
}fml_dataset;

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

void fml_layer_forward(fml_layer_header* layer_header, fml_data* prev_layer_activations);
void fml_layer_backprop(fml_layer_header* layer_header, fml_data* next_layer_gradient, fml_data* prev_layer_activations);
void fml_layer_learn(fml_layer_header* layer_header);

typedef struct{
  unsigned int n_layers;
  double learning_rate;
  fml_layer_header** layers;
}fml_net;


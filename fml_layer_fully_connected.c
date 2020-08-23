/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer_fully_connected.c * * * * * * * * * * * * * * * * * * * */
/* 22 august 2020  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_layer_fully_connected.h"

fml_layer_header* fml_layer_fully_connected_create(fml_data_shape* input, fml_data_shape* output){
  unsigned int input_size, output_size;
  fml_layer_fully_connected* ret = malloc(sizeof *ret);
  memcomment(ret, "fml_layer_fully_connected");
  fml_layer_header* header = (fml_layer_header*)ret;

  /* setup header */
  header->layer_type = FML_LAYER_TYPE_FULLY_CONNECTED;
  header->input_shape = input;
  memtest(output, "testing output in fml_layer_fully_connected");
  header->activation = fml_data_create(fml_data_shape_copy(output));
  memcomment(header->activation, "activation in fully connected layer");
  header->activation_gradient = fml_data_create(output);
  memcomment(header->activation_gradient, "activation gradient in fully connected layer");

  /* get layer size */
  memtest(input, "testing input in fml_layer_fully_connected");
  input_size = fml_data_shape_size(input);
  output_size = fml_data_shape_size(output);

  /* initialize weight matrix data */
  fml_data_shape* self_matrix_shape = fml_data_shape_create(output_size, input_size);
  memcomment(self_matrix_shape, "matrix shape for fully connected layer");
  ret->weight_matrix = fml_data_create(fml_data_shape_copy(self_matrix_shape));
  memcomment(ret->weight_matrix, "weight matrix for fully connected layer");
  ret->weight_matrix_gradient = fml_data_create(fml_data_shape_copy(self_matrix_shape));
  memcomment(ret->weight_matrix_gradient, "weight matrix gradient for fully connected layer");
  ret->weight_matrix_accumulated_gradient = fml_data_create(self_matrix_shape);
  memcomment(ret->weight_matrix_accumulated_gradient, "weight matrix gradient accumulator for fully connected layer");

  return header;
}

void fml_layer_fully_connected_input_forward(fml_layer_header* layer, fml_data* input){
  memtest(layer, "testing layer in fml_layer_fully_connected_input_forward");
  memtest(input, "testing input in fml_layer_fully_cpnnected_input_forward");

  /* cast header to fully connected layer pointer */
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;

  memtest(fully_connected->weight_matrix, "testing weight matrix in fml_layer_fully_connected_input_forward");
  memtest(layer->activation   , "testing activation in fml_layer_fully_connected_input_forward");

  /* perform matrix multiplication */
  fml_data_matrix_multiply(fully_connected->weight_matrix, input, layer->activation);
}

void fml_layer_fully_connected_forward(fml_layer_header* layer, fml_layer_header* prev_layer){
  memtest(layer, "testing layer in fml_layer_fully_connected_forward");
  memtest(prev_layer, "testing prev_layer in fml_layer_fully_connected_forward");
  
  /* cast header to fully connected layer */
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;

  memtest(fully_connected->weight_matrix, "testing weight matrix in fml_layer_fully_connected_forward");
  memtest(prev_layer->activation, "testing prev activation layer in fml_layer_fully_connected_forward");
  memtest(layer->activation, "testing current layer activation in fml_layer_fully_connected_forward");
  /* perform matrix multiplication */
  fml_data_matrix_multiply(fully_connected->weight_matrix, prev_layer->activation, layer->activation);
}

void fml_layer_fully_connected_backprop(fml_layer_header* layer, fml_layer_header* prev_layer){
  memtest(layer, "testing layer in fml_layer_fully_connected_backprop");
  memtest(prev_layer, "testing prev_layer in fml_layer_fully_connected_backprop");

  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;
  unsigned j, i;
  unsigned M, N;
  double tmp;
  fml_data *weight_matrix = fully_connected->weight_matrix;
  fml_data *curr_activation_gradient = layer->activation_gradient;
  fml_data *prev_activation_gradient = prev_layer->activation_gradient;
  fml_data *prev_activation = prev_layer->activation;
  fml_data *weight_matrix_gradient = fully_connected->weight_matrix_gradient;
  fml_data *weight_matrix_accumulated_gradient = fully_connected->weight_matrix_accumulated_gradient;

  memtest(weight_matrix, "testing weight matrix in fml_layer_fully_connected_backprop");
  memtest(curr_activation_gradient, "testing curr_activation_gradient in fml_layer_fully_connected_backprop");
  memtest(prev_activation_gradient, "testing prev_activation_gradient in fml_layer_fully_connected_backprop");
  memtest(prev_activation, "testing prev_activation in fml_layer_fully_connected_backprop");
  memtest(weight_matrix_gradient, "testing weight_matrix_gradient in fml_layer_fully_connected_backprop");
  memtest(weight_matrix_accumulated_gradient, "testing accumulated gradient in fml_layer_fully_connected_backprop");
  fml_data_get_dimensions(weight_matrix, 2, &M, &N);
  
  /* set previous layer activation gradient */
  /* dC/dx[i] = SUM_{j = 1}^{M}(W[j][i] * dC/dy[j])*/
  for(i = 0; i < N; ++i){
    tmp = 0.;
    for(j = 0; j < M; ++j){
      tmp += fml_data_subscript_get(weight_matrix, 2, j, i) * fml_data_subscript_get(curr_activation_gradient, 1, j);
    }
    fml_data_subscript_set(prev_activation_gradient, tmp, 1, i);
  }

  /* set weight matrix gradient */
  /* dC/dW[j][i] = x[i] * dC/dy[j] */
  for(j = 0; j < M; ++j){
    for(i = 0; i < N; ++i){
      tmp = fml_data_subscript_get(prev_activation, 1, i) * fml_data_subscript_get(curr_activation_gradient, 1, j);
      fml_data_subscript_set(weight_matrix_gradient, tmp, 2, j, i);
    }
  }

  /* accumulate gradient */
  fml_data_add(weight_matrix_accumulated_gradient, weight_matrix_gradient, weight_matrix_accumulated_gradient);
}

double 
fml_layer_fully_connected_output_backprop(fml_layer_header* layer, fml_layer_header* prev_layer, 
                                          fml_data* label, cost_function c)
{
  memtest(layer, "testing layer in fml_layer_fully_connected_output_backprop");
  memtest(label, "testing label in fml_layer_fully_connected_output_backprop");

  memtest(layer->activation, "testing layer->activation in fml_layer_fully_connected_output_backprop");
  memtest(layer->activation_gradient, "testing layer->activation_gradient in fml_layer_fully_connected_output_backprop");
  double cost = c(layer->activation, label, layer->activation_gradient);
  fml_layer_fully_connected_backprop(layer, prev_layer);

  return cost;
}

void fml_layer_fully_connected_input_backprop(fml_layer_header* layer, fml_data* input){
  memtest(layer, "testing layer in fml_layer_fully_connected_backprop");
  memtest(input, "testing input in fml_layer_fully_connected_backprop");

  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;
  unsigned j, i;
  unsigned M, N;
  double tmp;
  fml_data *weight_matrix = fully_connected->weight_matrix;
  fml_data *curr_activation_gradient = layer->activation_gradient;
  fml_data *weight_matrix_gradient = fully_connected->weight_matrix_gradient;
  fml_data *weight_matrix_accumulated_gradient = fully_connected->weight_matrix_accumulated_gradient;

  memtest(weight_matrix, "testing weight matrix in fml_layer_fully_connected_backprop");
  memtest(curr_activation_gradient, "testing curr_activation_gradient in fml_layer_fully_connected_backprop");
  memtest(weight_matrix_gradient, "testing weight_matrix_gradient in fml_layer_fully_connected_backprop");
  memtest(weight_matrix_accumulated_gradient, "testing accumulated gradient in fml_layer_fully_connected_backprop");
  fml_data_get_dimensions(weight_matrix, 2, &M, &N);

  /* set weight matrix gradient */
  /* dC/dW[j][i] = x[i] * dC/dy[j] */
  for(j = 0; j < M; ++j){
    for(i = 0; i < N; ++i){
      tmp = fml_data_subscript_get(input, 1, i) * fml_data_subscript_get(curr_activation_gradient, 1, j);
      fml_data_subscript_set(weight_matrix_gradient, tmp, 2, j, i);
    }
  }

  /* accumulate gradient */
  fml_data_add(weight_matrix_accumulated_gradient, weight_matrix_gradient, weight_matrix_accumulated_gradient);
}

void fml_layer_fully_connected_learn(fml_layer_header* layer, double rate){
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;
  memtest(layer, "testing layer in fml_layer_fully_connected_learn");
  fml_data *weight_matrix_accumulated_gradient = fully_connected->weight_matrix_accumulated_gradient;
  fml_data *weight_matrix = fully_connected->weight_matrix;

  memtest(weight_matrix_accumulated_gradient, "testing accumulated gradient in fml_layer_fully_connected_learn");
  fml_data_scale(weight_matrix_accumulated_gradient, rate);
  memtest(weight_matrix, "testing weight matrix in fml_layer_fully_connected");
  fml_data_add(weight_matrix_accumulated_gradient, weight_matrix, weight_matrix);
  fml_data_reset(weight_matrix_accumulated_gradient);
}

void fml_layer_fully_connected_destroy(fml_layer_header* layer){
  memtest(layer, "testing layer in fml_layer_fully_connected_destroy");
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;

  memtest(fully_connected->weight_matrix, "testing weight matrix in fml_layer_fully_connected-destroy");
  memtest(fully_connected->weight_matrix_gradient, "testing weight matrix gradient in fml_layer_fully_connected-destroy");
  memtest(fully_connected->weight_matrix_accumulated_gradient, "testing gradient accumulator in fml_layer_fully_connected-destroy");
  free(fully_connected->weight_matrix);
  free(fully_connected->weight_matrix_gradient);
  free(fully_connected->weight_matrix_accumulated_gradient);

  fml_layer_header_destroy_items(layer);
  free(layer);
}


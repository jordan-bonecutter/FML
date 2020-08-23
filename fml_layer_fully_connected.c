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

  header->layer_type = FML_LAYER_TYPE_FULLY_CONNECTED;
  header->input_shape = input;
  memtest(output, "testing output in fml_layer_fully_connected");
  header->activation = fml_data_create(fml_data_shape_copy(output));
  memcomment(header->activation, "activation in fully connected layer");
  header->activation_gradient = fml_data_create(output);
  memcomment(header->activation_gradient, "activation gradient in fully connected layer");

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

  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;

  memtest(fully_connected->weight_matrix, "testing weight matrix in fml_layer_fully_connected_input_forward");
  memtest(layer->activation   , "testing activation in fml_layer_fully_connected_input_forward");
  fml_data_matrix_multiply(fully_connected->weight_matrix, input, layer->activation);
}

void fml_layer_fully_connected_forward(fml_layer_header* layer, fml_layer_header* prev_layer){
  memtest(layer, "testing layer in fml_layer_fully_connected_forward");
  memtest(prev_layer, "testing prev_layer in fml_layer_fully_connected_forward");
  
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;

  memtest(fully_connected->weight_matrix, "testing weight matrix in fml_layer_fully_connected_forward");
  memtest(prev_layer->activation, "testing prev activation layer in fml_layer_fully_connected_forward");
  memtest(layer->activation, "testing current layer activation in fml_layer_fully_connected_forward");
  fml_data_matrix_multiply(fully_connected->weight_matrix, prev_layer->activation, layer->activation);
}

void fml_layer_fully_connected_backprop(fml_layer_header* layer, fml_layer_header* next_layer){
  memtest(layer, "testing layer in fml_layer_fully_connected_backprop");
  memtest(next_layer, "testing next_layer in fml_layer_fully_connected_backprop");

  unsigned i, j;
  unsigned curr_activation_size = fml_data_size(layer->activation);
  unsigned next_activation_size = fml_data_size(next_layer->activation);
  double tmp;
  fml_layer_fully_connected *fully_connected = (fml_layer_fully_connected*)layer;
  memtest(fully_connected->weight_matrix, "testing fully_connected->weight matrix in fml_layer_fully_connected_backprop");
  memtest(next_layer->activation_gradient, "testing next_layer->act_gradient in fml_layer_fully_connected_backprop");
  memtest(layer->activation_gradient, "testing layer->activation_gradeint in fml_layer_fully_connected_backprop");
  memtest(layer->activation, "testing layer->activation in fml_layer_fully_connected_backprop");
  memtest(fully_connected->weight_matrix_gradient, "testing weight matrix grad in fml_layer_fully_connected_backprop");
  memtest(fully_connected->weight_matrix_accumulated_gradient, "testing accumulator fml_layer_fully_connected_backprop");
  fml_data *weight_matrix = fully_connected->weight_matrix;
  fml_data *next_activation_gradient = next_layer->activation_gradient;
  fml_data *curr_activation_gradient = layer->activation_gradient;
  fml_data *curr_activation = layer->activation;
  fml_data *weight_matrix_gradient = fully_connected->weight_matrix_gradient;
  fml_data *weight_matrix_accumulated_gradient = fully_connected->weight_matrix_accumulated_gradient;

  /* set activation gradient */
  for(i = 0; i < curr_activation_size; ++i){
    tmp = 0.;
    for(j = 0; j < next_activation_size; ++j){
      tmp += fml_data_subscript_get(weight_matrix, j, i) * fml_data_subscript_get(next_activation_gradient, j);
    }
    fml_data_subscript_set(curr_activation_gradient, tmp, i);
  }

  /* set weight matrix gradient */
  for(i = 0; i < curr_activation_size; ++i){
    for(j = 0; j < next_activation_size; ++j){
      tmp = fml_data_subscript_get(curr_activation, i) * fml_data_subscript_get(next_activation_gradient, j);
      fml_data_subscript_set(weight_matrix_gradient, tmp, j, i);
    }
  }

  /* accumulate gradient */
  fml_data_add(weight_matrix_accumulated_gradient, weight_matrix_gradient, weight_matrix_accumulated_gradient);
}


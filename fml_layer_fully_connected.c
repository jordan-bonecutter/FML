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

  /* initialize bias vector data */
  fml_data_shape* self_bias_vector_shape = fml_data_shape_create(output_size, 1);
  memcomment(self_bias_vector_shape, "bias vector shape for fully connected layer");
  ret->bias_vector = fml_data_create(fml_data_shape_copy(self_bias_vector_shape));
  memcomment(ret->bias_vector, "bias vector for fully connected layer");
  ret->bias_vector_gradient = fml_data_create(fml_data_shape_copy(self_bias_vector_shape));
  memcomment(ret->bias_vector_gradient, "bias vector gradient for fully connected layer");
  ret->bias_vector_accumulated_gradient = fml_data_create(self_bias_vector_shape);
  memcomment(ret->bias_vector_accumulated_gradient, "bias vector gradient accumulator for fully connected layer");

  return header;
}


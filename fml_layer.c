/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer.c * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_internal.h"
#include <stdlib.h>

typedef struct{
  fml_layer_header header;
  fml_data* weight_matrix;
  fml_data* bias_vector;
  fml_data* weight_matrix_gradient;
  fml_data* bias_vector_gradient;
  fml_data* weight_matrix_accumulated_gradient;
  fml_data* bias_vector_accumulated_gradient;
}fml_layer_fully_connected;

typedef struct{ 
  fml_layer_header header;
  double alpha;
}fml_layer_leaky_relu;

typedef struct{
  fml_layer_header header;
  fml_data_shape* filter_size;
  unsigned int n_filters;
}fml_layer_convolution;

fml_layer_header* fml_layer_fully_connected_create(fml_data_shape* input, fml_data_shape* output){
  unsigned int input_size, output_size;
  fml_layer_fully_connected* ret = malloc(sizeof *ret);
  fml_layer_header* header = (fml_layer_header*)ret;
  
  header->layer_type = FML_LAYER_TYPE_FULLY_CONNECTED;
  header->input_shape = input;
  header->activations = fml_data_create(output);

  input_size = fml_data_shape_size(input);
  output_size = fml_data_shape_size(output);

  /* initialize weight matrix data */
  fml_data_shape* self_matrix_shape = fml_data_shape_create(output_size, input_size);
  ret->weight_matrix = fml_data_create(fml_data_shape_copy(self_matrix_shape));
  ret->weight_matrix_gradient = fml_data_create(fml_data_shape_copy(self_matrix_shape));
  ret->weight_matrix_accumulated_gradient = fml_data_create(self_matrix_shape);

  /* initialize bias vector data */
  fml_data_shape* self_bias_vector_shape = fml_data_shape_create(output_size, 1);
  ret->bias_vector = fml_data_create(fml_data_shape_copy(self_bias_vector_shape));
  ret->bias_vector_gradient = fml_data_create(fml_data_shape_copy(self_bias_vector_shape));
  ret->bias_vector_accumulated_gradient = fml_data_create(self_bias_vector_shape);

  return header;
}

#if 0
fml_layer* fml_layer_sigmoid_create(fml_data_shape* size);
fml_layer* fml_layer_tanh_create(fml_data_shape* size);
fml_layer* fml_layer_relu_create(fml_data_shape* size);
fml_layer* fml_layer_leaky_relu_create(fml_data_shape* size, double alpha);
fml_layer* fml_layer_batch_normalize(fml_data_shape* size);
fml_layer* fml_layer_convolution_create(fml_data_shape* input, fml_data_size* filter_size, unsigned int n_filters);
void       fml_layer_destroy(fml_layer* layer);
#endif


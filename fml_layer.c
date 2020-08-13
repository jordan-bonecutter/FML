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
  unsigned int n_filters;
  fml_data** filters;
  fml_data** filter_gradients;
  fml_data** filter_accumulated_gradients;
}fml_layer_convolution;

fml_layer_header* fml_layer_convolution_create(fml_data_shape* input, fml_data_shape* filter_size, unsigned int n_filters){
  fml_layer_convolution* ret = malloc(sizeof *ret);
  memcomment(ret, "convolutional layer");
  fml_layer_header* header = (fml_layer_header*)ret;
  unsigned filter_width, filter_height;
  fml_data_shape* output_shape;
  fml_data** filter_i;

  header->layer_type = FML_LAYER_TYPE_CONVOLUTION;
  memtest(input, "testing input in fml_layer_convolution_create");
  header->input_shape = input;
  
  /* calculate output size */
  memtest(filter_size, "testing filter_size in fml_layer_convolution_create");
  filter_height = fml_data_shape_get_dimension(filter_size, 0);
  filter_width = fml_data_shape_get_dimension(filter_size, 1);
  output_shape = fml_data_shape_create(filter_height, filter_width, n_filters);
  memcomment(output_shape, "output_layer in convolutional layer");
  header->activation = fml_data_create(fml_data_shape_copy(output_shape));
  memcomment(header->activation, "activation in convolutional layer")
  header->activation_gradient = fml_data_create(output_shape);
  memcomment(header->activation_gradient, "activation_gradient in convolutional layer")

  /* create filters and gradients */
  ret->n_filters = n_filters;
  ret->filters = malloc((sizeof *ret->filters)*n_filters*3);
  memcomment(ret->filters, "fml_data* filters base array in convolutional layer");
  ret->filter_gradients = ret->filters + n_filters;
  ret->filter_accumulated_gradients = ret->filters + (n_filters<<1);
  for(filter_i = ret->filters; filter_i != ret->filters + n_filters-1; ++filter_i){
    /* filter */
    *filter_i = fml_data_create(fml_data_shape_copy(filter_size));
    memcomment(*filter_i, "filter %d in convolutional layer", (filter_i - ret->filters)/(sizeof *filter_i));

    /* filter immediate gradient */
    *(filter_i + n_filters) = fml_data_create(fml_data_shape_copy(filter_size));
    memcomment(*(filter_i + n_filters), 
              "filter gradient %d in convolutional layer", (filter_i - ret->filters)/(sizeof *filter_i));

    /* filter gradient accumulator */
    *(filter_i + (n_filters<<1)) = fml_data_create(fml_data_shape_copy(filter_size));
    memcomment(*(filter_i + (n_filters<<1)), 
              "filter gradient accumulator %d in convolutional layer", 
              (filter_i - ret->filters)/(sizeof *filter_i));
  }
  *filter_i = fml_data_create(fml_data_shape_copy(filter_size));
  memcomment(*filter_i, "filter %d in convolutional layer", n_filters - 1);

  *(filter_i + n_filters) = fml_data_create(fml_data_shape_copy(filter_size));
  memcomment(*(filter_i + n_filters), "filter %d in convolutional layer", n_filters - 1);

  *(filter_i + (n_filters<<1)) = fml_data_create(filter_size);
  memcomment(*(filter_i + (n_filters<<1)), "filter %d in convolutional layer", n_filters - 1);

  return header;
}

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

fml_layer_header* fml_layer_sigmoid_create(fml_data_shape* shape){
  fml_layer_header* ret = malloc((sizeof *ret));
  memcomment(ret, "fml_layer_header for sigmoid layer");

  ret->layer_type = FML_LAYER_TYPE_SIGMOID;
  memtest(shape, "tetsing shape in fml_layer_sigmoid_create");
  ret->input_shape = shape;
  ret->activation = fml_data_create(fml_data_shape_copy(shape));
  memcomment(ret->activation, "activation for sigmoid layer");
  ret->activation_gradient = fml_data_create(fml_data_shape_copy(shape));
  memcomment(ret->activation_gradient, "activation gradient for sigmoid layer");

  return ret; 
}

#if 0
fml_layer* fml_layer_tanh_create(fml_data_shape* size);
fml_layer* fml_layer_relu_create(fml_data_shape* size);
fml_layer* fml_layer_leaky_relu_create(fml_data_shape* size, double alpha);
fml_layer* fml_layer_batch_normalize(fml_data_shape* size);
fml_layer* fml_layer_convolution_create(fml_data_shape* input, fml_data_size* filter_size, unsigned int n_filters);
void       fml_layer_destroy(fml_layer* layer);
#endif


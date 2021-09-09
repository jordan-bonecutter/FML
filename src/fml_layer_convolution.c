/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer_convolution.c * * * * * * * * * * * * * * * * * * * * * */
/* 21 august 2020  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_layer_convolution.h"

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
  memcomment(header->activation, "activation in convolutional layer");
  header->activation_gradient = fml_data_create(output_shape);
  memcomment(header->activation_gradient, "activation_gradient in convolutional layer");

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


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer_convolution.h * * * * * * * * * * * * * * * * * * * * * */
/* 21 august 2020  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_LAYER_CONVOLUTION_INCLUDE_GUARD
#define FML_LAYER_CONVOLUTION_INCLUDE_GUARD

#include "fml_layer_internal.h"

typedef struct{
  fml_layer_header header;
  unsigned int n_filters;
  fml_data** filters;
  fml_data** filter_gradients;
  fml_data** filter_accumulated_gradients;
}fml_layer_convolution;

void   fml_layer_input_convolution_forward(fml_layer_header* layer, fml_data* input);
void   fml_layer_convolution_forward(fml_layer_header* layer, fml_layer_header* prev_layer);
void   fml_layer_convolution_backprop(fml_layer_header* layer, fml_layer_header* prev_layer);
double fml_layer_output_convolution_backprop(fml_layer_header* layer, fml_layer_header* prev_layer, fml_data* label, cost_function c);
void   fml_layer_convolution_input_backprop(fml_layer_header* layer, fml_data* input);
void   fml_layer_convolution_learn(fml_layer_header* layer, double rate);
void   fml_layer_convolution_destroy(fml_layer_header* layer);

#endif


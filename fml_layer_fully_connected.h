/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer_fully_connected.h * * * * * * * * * * * * * * * * * * * */
/* 22 august 2020  * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_LAYER_FULLY_CONNECTED_H
#define FML_LAYER_FULLY_CONNECTED_H

#include "fml_layer_internal.h"

typedef struct{
  fml_layer_header header;
  fml_data* weight_matrix;
  fml_data* weight_matrix_gradient;
  fml_data* weight_matrix_accumulated_gradient;
}fml_layer_fully_connected;

fml_layer_header* fml_layer_fully_connected_create(fml_data_shape* input, fml_data_shape* output);
void   fml_layer_fully_connected_input_forward(fml_layer_header* layer, fml_data* input);
void   fml_layer_fully_connected_forward(fml_layer_header* layer, fml_layer_header* prev_layer);
void   fml_layer_fully_connected_backprop(fml_layer_header* layer, fml_layer_header* prev_layer);
double fml_layer_fully_connected_output_backprop(fml_layer_header* layer, fml_layer_header *prev_layer, fml_data* label, cost_function c);
void   fml_layer_fully_connected_input_backprop(fml_layer_header* layer, fml_data* input);
void   fml_layer_fully_connected_learn(fml_layer_header* layer, double rate);
void   fml_layer_fully_connected_destroy(fml_layer_header* layer);

#endif


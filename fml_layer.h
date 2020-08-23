/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer.h * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_LAYER_INCLUDE_GUARD
#define FML_LAYER_INCLUDE_GUARD

#include "fml_standard.h"
#include "fml_data.h"

typedef void fml_layer;

void   fml_layer_input_forward(fml_layer* layer, fml_data* input);
void   fml_layer_forward(fml_layer* layer, fml_layer* prev_layer);
void   fml_layer_backprop(fml_layer* layer, fml_layer* prev_layer);
double fml_layer_output_backprop(fml_layer* layer, fml_layer *prev_layer, fml_data* label, cost_function c);
void   fml_layer_input_backprop(fml_layer* layer, fml_data *input);
void   fml_layer_learn(fml_layer* layer, double rate);
void   fml_layer_destroy(fml_layer* layer);

#endif


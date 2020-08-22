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
  fml_data* bias_vector;
  fml_data* weight_matrix_gradient;
  fml_data* bias_vector_gradient;
  fml_data* weight_matrix_accumulated_gradient;
  fml_data* bias_vector_accumulated_gradient;
}fml_layer_fully_connected;

#endif


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer_internal.h  * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_LAYER_INTERNAL_INCLUDE_GUARD
#define FML_LAYER_INTERNAL_INCLUDE_GUARD

#include "fml_standard.h"
#include "fml_data.h"

typedef enum {
  FML_LAYER_TYPE_FULLY_CONNECTED,
  FML_LAYER_TYPE_SIGMOID,
  FML_LAYER_TYPE_TANH,
  FML_LAYER_TYPE_RELU,
  FML_LAYER_TYPE_LEAKY_RELU,
  FML_LAYER_TYPE_BATCH_NORMALIZE,
  FML_LAYER_TYPE_CONVOLUTION,
  FML_LAYER_TYPE_COUNT
} fml_layer_type;/* neural network layer type enum */

typedef struct{
  fml_layer_type layer_type;
  fml_data_shape* input_shape;
  fml_data* activation;
  fml_data* activation_gradient;
}fml_layer_header;

void fml_layer_header_destroy_items(fml_layer_header *header);

#endif


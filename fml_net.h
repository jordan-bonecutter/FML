/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_net.h * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_NET_INCLUDE_GUARD
#define FML_NET_INCLUDE_GUARD

#include "fml_standard.h"
#include "fml_layer.h"
#include "fml_data.h"

typedef enum{
  FML_NET_COST_TYPE_QUADRATIC,
  FML_NET_COST_TYPE_COUNT
}fml_net_cost_type;

typedef enum{
  FML_NET_REGULARIZATION_TYPE_NONE,
  FML_NET_REGULARIZATION_TYPE_L1,
  FML_NET_REGULARIZATION_TYPE_L2,
  FML_NET_REGULARIZATION_TYPE_COUNT
}fml_net_regularization_type;

typedef struct{
  fml_net_cost_type cost_type;
  fml_net_regularization_type regularization_type;
  unsigned int n_layers;
  double learning_rate, regularization_weight;
  fml_layer** layers;
}fml_net;

#endif


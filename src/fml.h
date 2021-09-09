/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml.h (fun machine learning)  * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_H_INCLUDE_GUARD
#define FML_H_INCLUDE_GUARD

#define user_owned
#define object_owned

#include <stdarg.h>
#include <stdio.h>

#include "fml_layer.h"
#include "fml_net.h"
#include "fml_data.h"

#if 0
typedef void fml_net;/* neural network structure for fml */
typedef void fml_layer;/* neural network layer for fml */
typedef void fml_data;/* data (similar to np.ndarray except type is always float64) */
typedef void fml_dataset;/* dataset (array of data) */
typedef void fml_data_shape;/* shape of fml_data */

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

fml_net* fml_net_create(...);/* allocate new neural network w/ specified layers */
void     fml_net_set_learning_rate(fml_net* net, double learning_rate);/* set learning rate for neural network */
double   fml_net_get_learning_rate(fml_net*);/* get learning rate for neural network */
void     fml_net_set_cost_type(fml_net* net, fml_net_cost_type t);
void     fml_net_set_regularization_type(fml_net* net, fml_net_regularization_type t, double weight);
void     fml_train(fml_net* net, user_owned fml_dataset* dataset, unsigned int epochs, unsigned int minibatch_size);
void     fml_net_destroy(fml_net* net);
void     fml_net_dump(fml_net* net, user_owned FILE* file);
fml_net* fml_net_parse(user_owned FILE* file);

fml_dataset* fml_dataset_create(object_owned fml_data* samples, object_owned fml_data* labels, unsigned int total, unsigned int n_train, unsigned int n_validate, unsigned int n_test);
fml_dataset* fml_dataset_destroy(fml_dataset* dataset, bool destroy_data);

fml_data_shape* fml_data_shape_create(unsigned int n, ...)
void            fml_data_shape_destroy(fml_data_shape* shape);
fml_data_shape* fml_data_shape_copy(fml_data_shape* data);
unsigned int    fml_data_shape_get_dimension(fml_data_shape* data, unsigned int dim);

fml_data* fml_data_create(object_owned fml_data_shape* shape);
fml_data* fml_data_create_with_data(object_owned fml_data_shape* shape, object_owned double* data);
void      fml_data_destroy(fml_data* data);

fml_layer* fml_layer_fully_connected_create(object_owned fml_data_shape* input, object_owned fml_data_shape* output);
fml_layer* fml_layer_sigmoid_create(object_owned object_owned fml_data_shape* size);
fml_layer* fml_layer_tanh_create(object_owned fml_data_shape* size);
fml_layer* fml_layer_relu_create(object_owned fml_data_shape* size);
fml_layer* fml_layer_leaky_relu_create(object_owned fml_data_shape* size, double alpha);
fml_layer* fml_layer_batch_normalize(object_owned fml_data_shape* size);
fml_layer* fml_layer_convolution_create(object_owned fml_data_shape* input, object_owned fml_data_shape* filter_size, unsigned int n_filters);
void       fml_layer_destroy(fml_layer* layer);
#endif

#endif


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml.h (fun machine learning)  * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_H_INCLUDE_GUARD
#define FML_H_INCLUDE_GUARD

#include <stdarg.h>
#include <stdio.h>

typedef void fml_net;/* neural network structure for fml */
typedef void fml_layer;/* neural network layer for fml */
typedef void fml_data;/* data (similar to np.ndarray except type is always float64) */
typedef void fml_dataset;/* dataset (array of data) */
typedef void fml_data_shape;/* shape of fml_data */

fml_net* fml_net_create(unsigned int n, ...);/* allocate new neural network w/ specified layers */
void     fml_net_set_learning_rate(fml_net* net, double learning_rate);/* set learning rate for neural network */
double   fml_net_get_learning_rate(fml_net*);/* get learning rate for neural network */
void     fml_train(fml_net* net, fml_dataset* dataset, unsigned int epochs, unsigned int minibatch_size);
void     fml_net_destroy(fml_net* net);
void     fml_net_dump(fml_net* net, FILE* file);
fml_net* fml_net_parse(FILE* file);

fml_dataset* fml_dataset_create(fml_data* data, unsigned int total, unsigned int n_train, unsigned int n_validate, unsigned int n_test);
fml_dataset* fml_dataset_destroy(fml_dataset* dataset, bool destroy_data);

fml_data_shape* fml_data_shape_create(unsigned int n, ...)
void            fml_data_shape_destroy(fml_data_shape* shape);

fml_data* fml_data_create(fml_data_shape* shape, double* data);
void      fml_data_destroy(fml_data* data);
double    fml_data_get(fml_data* data, unsigned int n, ...);
void      fml_data_set(fml_data* data, double set, unsigned int n, ...);

fml_layer* fml_layer_fully_connected_create(fml_data_shape* input, fml_data_size* output);
fml_layer* fml_layer_signoid_create(fml_data_shape* size);
fml_layer* fml_layer_tanh_create(fml_data_shape* size);
fml_layer* fml_layer_relu_create(fml_data_shape* size);
fml_layer* fml_layer_leaky_relu_create(fml_data_shape* size, double alpha);
fml_layer* fml_layer_batch_normalize(fml_data_shape* size);
fml_layer* fml_layer_convolution_create(fml_data_shape* input, fml_data_size* filter_size, unsigned int n_filters);

#endif


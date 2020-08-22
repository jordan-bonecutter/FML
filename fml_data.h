/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_data.h  * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_DATA_INCLUDE_GUARD
#define FML_DATA_INCLUDE_GUARD

#include "fml_standard.h"

typedef struct{
  unsigned int n_dimensions;
  unsigned int* dimension;
}fml_data_shape;

typedef struct{
  fml_data_shape* shape;
  double* data;
}fml_data;

typedef void (*cost_function)(fml_data*, fml_data*, fml_data*);

fml_data* fml_data_create(fml_data_shape* shape);
fml_data* fml_data_create_with_data(fml_data_shape* shape, double* data);

typedef struct{
  unsigned int n_data, n_train, n_validate, n_test;
  fml_data* samples;
  fml_data* labels;
}fml_dataset;

fml_data_shape* fml_data_shape_create(unsigned int d0, ...);
fml_data_shape* fml_data_shape_copy(fml_data_shape* shape);
bool fml_data_have_same_shape(fml_data* d1, fml_data* d2);
unsigned int fml_data_shape_size(fml_data_shape* data);
unsigned int fml_data_shape_get_dimension(fml_data_shape* data, unsigned int dim);

#endif

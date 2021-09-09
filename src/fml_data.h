/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_data.h  * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FML_DATA_INCLUDE_GUARD
#define FML_DATA_INCLUDE_GUARD

#include "fml_standard.h"
#include <stdarg.h>

typedef struct{
  unsigned int n_dimensions;
  unsigned int* dimension;
}fml_data_shape;

typedef struct{
  fml_data_shape* shape;
  double* data;
}fml_data;

typedef double (*cost_function)(fml_data *output, fml_data *label, fml_data *gradient);

fml_data* fml_data_create(fml_data_shape* shape);
fml_data* fml_data_create_with_data(fml_data_shape* shape, double* data);
unsigned int fml_data_size(fml_data *data);
void fml_data_get_dimensions(fml_data *data, unsigned dimensions, ...);

typedef struct{
  unsigned int n_data, n_train, n_validate, n_test;
  fml_data* samples;
  fml_data* labels;
}fml_dataset;

fml_data_shape* fml_data_shape_create(unsigned dimaensions, ...);
fml_data_shape* fml_data_shape_copy(fml_data_shape* shape);
bool fml_data_have_same_shape(fml_data* d1, fml_data* d2);
unsigned int fml_data_shape_size(fml_data_shape* data);
unsigned int fml_data_shape_get_dimension(fml_data_shape* data, unsigned int dim);

/* C := A*B */
void fml_data_matrix_multiply(fml_data *A, fml_data *B, fml_data *C);

/* C := A + B*/
void fml_data_add(fml_data *A, fml_data *B, fml_data *C);

/* A = sA */
void fml_data_scale(fml_data *A, double scale);

/* A = {{...{0, 0, ...}...}, ...} */
void fml_data_reset(fml_data *A);

double fml_data_subscript_get(fml_data *data, int n, ...);
void fml_data_subscript_set(fml_data *data, double val, int n, ...);
void fml_data_destroy(fml_data *A);

#endif


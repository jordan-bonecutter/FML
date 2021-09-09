/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_data.c  * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 8 september 2021  * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "fml_data.h"
#include <assert.h>
#include <stdlib.h>

/*
typedef struct{
  unsigned int n_dimensions;
  unsigned int* dimension;
}fml_data_shape;

typedef struct{
  fml_data_shape shape;
  double* data;
}fml_data;

typedef double (*cost_function)(fml_data *output, fml_data *label, fml_data *gradient);
*/

fml_data* fml_data_create(fml_data_shape* shape) {
  fml_data *ret;
  size_t total_data_size;

  memtest(shape, "testing input shape in fml_data_create");

  // Calculate total data size as product of all dimensions
  total_data_size = 1;
  unsigned int *head = shape->dimension;
  for(unsigned int i = 0; i < shape->n_dimensions; ++i, ++head) {
    memtest(head, "Testing fml_data_shape[%d] with length %d", i, shape->n_dimensions);
    total_data_size *= ((size_t)*head);
  }

  ret = malloc(sizeof(fml_data) + total_data_size);
  memcomment(ret, "fml_data in fml_data_create");
  ret->shape = *shape;

  return ret;
}

#if 0
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


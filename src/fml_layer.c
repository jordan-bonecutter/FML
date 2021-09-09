/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* fml_layer.c * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 6 august 2020 * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* jordan bonecutter * * * * * * * * * * * * * * * * * * * * * * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdlib.h>

/*
typedef struct{ 
  fml_layer_header header;
  double alpha;
}fml_layer_leaky_relu;

void fml_first_layer_forward(fml_layer_header* layer, fml_data* input){
  memtest(layer, "testing layer in fml_first_layer_forward");
  memtest(layer, "testing input in fml_first_layer_forward");
  switch(){
  }
}

fml_layer_header* fml_layer_sigmoid_create(fml_data_shape* shape){
  fml_layer_header* ret = malloc((sizeof *ret));
  memcomment(ret, "fml_layer_header for sigmoid layer");

  ret->layer_type = FML_LAYER_TYPE_SIGMOID;
  memtest(shape, "tetsing shape in fml_layer_sigmoid_create");
  ret->input_shape = shape;
  ret->activation = fml_data_create(fml_data_shape_copy(shape));
  memcomment(ret->activation, "activation for sigmoid layer");
  ret->activation_gradient = fml_data_create(fml_data_shape_copy(shape));
  memcomment(ret->activation_gradient, "activation gradient for sigmoid layer");

  return ret; 
}*/

#if 0
fml_layer* fml_layer_tanh_create(fml_data_shape* size);
fml_layer* fml_layer_relu_create(fml_data_shape* size);
fml_layer* fml_layer_leaky_relu_create(fml_data_shape* size, double alpha);
fml_layer* fml_layer_batch_normalize(fml_data_shape* size);
fml_layer* fml_layer_convolution_create(fml_data_shape* input, fml_data_size* filter_size, unsigned int n_filters);
void       fml_layer_destroy(fml_layer* layer);
#endif


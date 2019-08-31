#ifndef QUANT_CONVOLUTIONAL_LAYER_H
#define QUANT_CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer quant_convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

quant_convolutional_layer make_quant_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_quant_convolutional_layer(quant_convolutional_layer *layer, int w, int h);
void forward_quant_convolutional_layer(const quant_convolutional_layer layer, network net);
void update_quant_convolutional_layer(quant_convolutional_layer layer, update_args a);
void backward_quant_convolutional_layer(quant_convolutional_layer layer, network net);

int quant_convolutional_out_height(quant_convolutional_layer layer);
int quant_convolutional_out_width(quant_convolutional_layer layer);

#endif


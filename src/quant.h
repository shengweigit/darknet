#ifndef QUANT_H_
#define QUANT_H_

#include "darknet.h"

int quant_dbg;
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
unsigned char cast_to_uint8(float input);
void tensor_min_max(float *input, int total, float *min, float *max);
void update_fmin_max(fmin_max_t *fmin_max, float fmin, float fmax);
void fake_quant_with_min_max(float *input, int total, float real_min, float real_max, int num_bits, float *output);
void choose_quant_param(float real_min, float real_max, int num_bits, double *scale, int *zero_point);
void chose_multipler(double input1_scale, double input2_scale, double output_scale, double *multiplier);
void chose_convolution_multipler(double input_scale, double weight_scale, double bias_scale, double output_scale, double *multiplier);
void chose_activation_range(double output_scale, int output_zero_point, int num_bits, int *quant_act_min, int *quant_act_max);
void quantize_uint8(float *input, int total, double scale, int zero_point, unsigned char *output);
void dequantize_uint8(unsigned char *input, int total, double scale, int zero_point, float *output);
void quantize_int32(float *input, int total, double scale, int zero_point, int *output);
void dequantize_int32(int *input, int total, double scale, int zero_point, float *output);
void quantize_and_dequantize_uint8(float *input, int total, double scale, int zero_point);
int offset(int d0, int d1, int d2, int d3, int i0, int i1, int i2, int i3);
void fold_batchnorm(float *weights, int cout, int cin, int size, float *means, float *vars, float *gammas, float *betas, float *out_weights, float *out_biases);
#endif /* QUANT_H_ */

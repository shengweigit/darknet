#include <stdio.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include "quant.h"
#include <limits.h>

int quant_dbg = 1;

unsigned char cast_to_uint8(float input)
{
	return MIN(MAX(0.0, input), 255.0);
}

int cast_to_int32(float input)
{
	return MIN(MAX(INT_MIN, input), INT_MAX);
}

void tensor_min_max(float *input, int total, float *min, float *max)
{
	float fmin = FLT_MAX;
	float fmax = FLT_MIN;
    for(int i = 0; i < total; ++i){
		fmin = MIN(input[i], fmin);
		fmax = MAX(input[i], fmax);
    }
    *min = fmin;
    *max = fmax;
}

void update_fmin_max(fmin_max_t *fmin_max, float fmin, float fmax)
{
	if(fmin < fmin_max->min) {
		//fprintf(stderr, "update fmin %.6f-->%.6f\n", fmin_max->min, fmin);
		fmin_max->min = fmin;
	}
	if(fmax > fmin_max->max) {
		//fprintf(stderr, "update fmax %.6f-->%.6f\n", fmin_max->max, fmax);
		fmin_max->max = fmax;
	}
}




static void nudge(float real_min, float real_max, int quant_min, int quant_max, float *nudged_min, float *nudged_max, float *nudged_scale)
{
	float quant_min_float = quant_min;
	float quant_max_float = quant_max;
	float scale, zero_point_from_min;
	unsigned short nudged_zero_point;

	scale = (real_max - real_min) / (quant_max_float - quant_min_float);
	zero_point_from_min = quant_min_float - real_min / scale;
	if (zero_point_from_min < quant_min_float) {
		nudged_zero_point = quant_min;
	} else if (zero_point_from_min > quant_max_float) {
		nudged_zero_point = quant_max;
	} else {
		nudged_zero_point = round(zero_point_from_min);
	}

	*nudged_min = (quant_min_float - nudged_zero_point) * scale;
	*nudged_max = (quant_max_float - nudged_zero_point) * scale;
	*nudged_scale = scale;
}

void fake_quant_with_min_max(float *input, int total, float real_min, float real_max, int num_bits, float *output)
{
	assert(real_min <= 0.0f && real_max >= 0.0f);
	assert(real_min < real_max);

	int quant_min, quant_max;
	quant_min = 0;
	quant_max = (1 << num_bits) - 1;

	float nudged_min, nudged_max, nudged_scale;
	nudge(real_min, real_max, quant_min, quant_max, &nudged_min, &nudged_max, &nudged_scale);

	float clamped, clamped_shifted;
	float inv_nudged_scale = 1.0f / nudged_scale;
	float original, error = 0.0f;
	for (int i=0; i < total; i++) {
		original = input[i];
		clamped = MAX(original, nudged_min);
		clamped = MIN(clamped, nudged_max);
		clamped_shifted = clamped - nudged_min;
		output[i] = floor((clamped_shifted * inv_nudged_scale + 0.5f)) * nudged_scale + nudged_min;
		error = error + fabs(original - output[i]);
		if (i == 0) {
			fprintf(stderr, "real: %.6f, quant: %.6f\n", original, output[i]);
		}
	}
	fprintf(stderr, "avg quant error: %.6f\n", error / total);
}

// zero_point = quant_min - real_min / scale
void choose_quant_param(float real_min, float real_max, int num_bits, double *scale, int *zero_point)
{
	assert(real_min <= 0.0f && real_max >= 0.0f);
	assert(real_min < real_max);

	int quant_min, quant_max;
	quant_min = 0;
	quant_max = (1 << num_bits) - 1;
	double quant_min_float = quant_min;
	double quant_max_float = quant_max;
	if (real_min == real_max) {
		*scale = 0.0f;
		*zero_point = 0;
		return;
	}

	double tmp_scale = (real_max - real_min) / (quant_max_float - quant_min_float);
	double zero_point_from_min = quant_min_float - real_min / tmp_scale;
	double zero_point_from_max = quant_max_float - real_max / tmp_scale;
	double zero_point_from_min_error = abs(quant_min_float) + abs(real_min / tmp_scale);
	double zero_point_from_max_error = abs(quant_max_float) + abs(real_max / tmp_scale);
	double zero_point_float =
			zero_point_from_min_error < zero_point_from_max_error
				? zero_point_from_min
				: zero_point_from_max;

	int nudged_zero_point = 0;
	if (zero_point_float < quant_min_float) {
		nudged_zero_point = quant_min;
	} else if (zero_point_from_min > quant_max_float) {
		nudged_zero_point = quant_max;
	} else {
		nudged_zero_point = round(zero_point_from_min);
	}
	assert(nudged_zero_point >= quant_min);
	assert(nudged_zero_point <= quant_max);

	*scale = tmp_scale;
	*zero_point = nudged_zero_point;
}

void chose_multipler(double input1_scale, double input2_scale, double output_scale, double *multiplier)
{
	double input_product_scale = input1_scale * input2_scale;
//	assert(abs(input_product_scale - bias_scale) <= 1e-6f * MIN(input_scale, bias_scale));
//	assert(input_product_scale >= 0);
	*multiplier = input_product_scale / output_scale;
}

void chose_activation_range(double output_scale, int output_zero_point, int num_bits, int *quant_act_min, int *quant_act_max)
{
	*quant_act_min = output_zero_point;
	*quant_act_max = (1 << num_bits) - 1;
}

// quant_val = real_val * (1.0 / scale) + zero_point
void quantize_uint8(float *input, int total, double scale, int zero_point, unsigned char *output)
{
	double inverse_scale = 1.0f / scale;
	double scaled_val;
	for (int i=0; i<total; i++) {
		scaled_val = input[i] * inverse_scale + zero_point;
		output[i] = cast_to_uint8(round(scaled_val));
	}
}

// real_val = scale * (quant_val - zero_point)
void dequantize_uint8(unsigned char *input, int total, double scale, int zero_point, float *output)
{
	double input_val;
	for (int i=0; i<total; i++) {
		input_val = input[i];
		output[i] = scale * (input_val - zero_point);
	}
}

void quantize_int32(float *input, int total, double scale, int zero_point, int *output)
{
	double inverse_scale = 1.0f / scale;
	double scaled_val;
	for (int i=0; i<total; i++) {
		scaled_val = input[i] * inverse_scale + zero_point;
		output[i] = cast_to_int32(round(scaled_val));
	}
}

void dequantize_int32(int *input, int total, double scale, int zero_point, float *output)
{
	double input_val;
	for (int i=0; i<total; i++) {
		input_val = input[i];
		output[i] = scale * (input_val - zero_point);
	}
}

void quantize_and_dequantize_uint8(float *input, int total, double scale, int zero_point)
{
	unsigned char *quant_output = calloc(total, sizeof(unsigned char));
	float *real_output = calloc(total, sizeof(float));
	quantize_uint8(input, total, scale, zero_point, quant_output);
	dequantize_uint8(quant_output, total, scale, zero_point, real_output);

	float error = 0.0f;
	for (int i=0; i<total; i++) {
		error = error + fabs(input[i] - real_output[i]);
		if (i == 0) {
			fprintf(stderr, "raw: %.6f, qu/de: %.6f\n", input[i], real_output[i]);
		}
	}
	fprintf(stderr, "avg quant error: %.6f\n", error / total);

	free(quant_output);
	free(real_output);
}

int offset(int d0, int d1, int d2, int d3, int i0, int i1, int i2, int i3) {
	assert(i0 >= 0 && i0 < d0);
	assert(i1 >= 0 && i1 < d1);
	assert(i2 >= 0 && i2 < d2);
	assert(i3 >= 0 && i3 < d3);
	return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

void fold_batchnorm(float *weights, int cout, int cin, int size,
		float *means, float *vars, float *gammas, float *betas,
		float *out_weights, float *out_biases)
{
	float *ptr_weights = weights;
	float *ptr_out_weights = out_weights;
	float *ptr_biases = out_biases;

	int spatial = size * size;
	float epsilon = 0.000001f;
	for (int o=0; o<cout; o++) {
		float inverse_scale = 1.0f / sqrt(vars[o] + epsilon);
		float gamma = gammas[o];
		for (int i=0; i<cin; i++) {
			for (int s=0; s<spatial; s++) {
				*ptr_out_weights = *ptr_weights * gamma * inverse_scale;
				++ptr_weights;
				++ptr_out_weights;
			}
		}
	}
	for (int o=0; o<cout; o++) {
		float inverse_scale = 1.0f / sqrt(vars[o] + epsilon);
		float gamma = gammas[o];
		float beta = betas[o];
		float mean = means[o];
		*ptr_biases = beta - gamma * mean * inverse_scale;
		++ptr_biases;
	}

	float min, max;
	tensor_min_max(out_weights, cout*cin*size*size, &min, &max);
	fprintf(stderr, "[fold_batchnorm] weight min: %.6f, weight max: %.6f\n", min, max);
	tensor_min_max(out_biases, cout, &min, &max);
	fprintf(stderr, "[fold_batchnorm] bias min: %.6f, bias max: %.6f\n", min, max);
}

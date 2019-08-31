#include "quant_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include "quant.h"
#include <limits.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

static void swap_binary(quant_convolutional_layer *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

static void binarize_weights(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

static void binarize_cpu(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

static void binarize_input(float *input, int n, int size, float *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int quant_convolutional_out_height(quant_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int quant_convolutional_out_width(quant_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

quant_convolutional_layer make_quant_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    int i;
    quant_convolutional_layer l = {0};
    l.type = QUANT_CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    int out_w = quant_convolutional_out_width(l);
    int out_h = quant_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_quant_convolutional_layer;
    l.backward = backward_quant_convolutional_layer;
    l.update = update_quant_convolutional_layer;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }
    l.fmin_max = calloc(2, sizeof(fmin_max_t));
    l.quant_param = calloc(2, sizeof(quant_param_t));
    l.output_fmin_max = calloc(1, sizeof(fmin_max_t));
    l.output_quant_param = calloc(1, sizeof(quant_param_t));

    l.quant_weight = calloc(c/groups*n*size*size, sizeof(unsigned char));
    l.quant_input =calloc(l.c*l.h*l.w*l.batch, sizeof(unsigned char));
    l.quant_output = calloc(l.batch*l.out_c*l.out_h*l.out_w, sizeof(unsigned char));
    l.quant_biases = calloc(n, sizeof(int));

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if(binary){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if(xnor){
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
        }

        if(batch_normalize){
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean, n);
            l.variance_delta_gpu = cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);

    return l;
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_quant_convolutional_layer(quant_convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = quant_convolutional_out_width(*l);
    int out_h = quant_convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

static void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

//static void scale_bias(float *output, float *scales, int batch, int n, int size)
//{
//    int i,j,b;
//    for(b = 0; b < batch; ++b){
//        for(i = 0; i < n; ++i){
//            for(j = 0; j < size; ++j){
//                output[(b*n + i)*size + j] *= scales[i];
//            }
//        }
//    }
//}

static void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_quant_convolutional_layer(quant_convolutional_layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    //if (!net.input_quant_param) {
    	float fmin, fmax;
    	net.input_quant_param = calloc(1, sizeof(quant_param_t));
    	//assert(net.input_quant_param);
    	tensor_min_max(net.input, l.c*l.h*l.w*l.batch, &fmin, &fmax);
    	choose_quant_param(fmin, fmax, 8,
    			&net.input_quant_param[0].scale, &net.input_quant_param[0].zero_point);
    	fprintf(stderr, "[online quant input] min: %.6f, max: %.6f\n", fmin, fmax);
    //}
	const unsigned char *input_data = l.quant_input;
	const unsigned char *filter_data = l.quant_weight;
	unsigned char *output_data = l.quant_output;
	const int *bias_data = l.quant_biases;
	const int input_offset =  -net.input_quant_param[0].zero_point;
	const int filter_offset = -l.quant_param[0].zero_point;
	const int conv_output_offset = l.quant_param[1].zero_point;
	const int activation_output_offset = l.output_quant_param[0].zero_point;
	int leaky_factor_offset;
	const double input_scale = net.input_quant_param[0].scale;
	const double filter_scale = l.quant_param[0].scale;
	const double bias_scale = input_scale * filter_scale; 	// force to bias scale = input_scale * weight_scale, zero_point = 0;
	const double conv_output_scale = l.quant_param[1].scale;
	const double activation_output_scale = l.output_quant_param[0].scale;
	double leaky_factor_scale;
	double conv_output_multiplier;
	double activation_output_multiplier;
	const int output_shift = 0;
	int output_activation_min;
	int output_activation_max;
	float leaky_factor = 0.1f;
	unsigned char quant_leaky_factor;

	choose_quant_param(0.0f, leaky_factor, 8, &leaky_factor_scale, &leaky_factor_offset);
	quantize_uint8(&leaky_factor, 1, leaky_factor_scale, leaky_factor_offset, &quant_leaky_factor);

	quantize_uint8(net.input, l.c*l.h*l.w*l.batch, net.input_quant_param[0].scale, net.input_quant_param[0].zero_point, input_data);
	quantize_int32(l.biases, l.out_c, bias_scale, 0, bias_data);

	chose_multipler(input_scale, filter_scale, conv_output_scale, &conv_output_multiplier);
	chose_multipler(conv_output_scale, leaky_factor_scale, activation_output_scale, &activation_output_multiplier);



	// input:  n c h w
	// weight: co ci h w
	// output: n c h w
	for (int batch = 0; batch < l.batch; ++batch) {
		for (int out_y = 0; out_y < l.out_h; ++out_y) {
			for (int out_x = 0; out_x < l.out_w; ++out_x) {
				for (int out_channel = 0; out_channel < l.out_c; ++out_channel) {
					const int in_x_origin = (out_x * l.stride) - l.pad;
					const int in_y_origin = (out_y * l.stride) - l.pad;
					double acc = 0.0f;
					for (int in_channel = 0; in_channel < l.c; ++in_channel) {
						for (int filter_y = 0; filter_y < l.size; ++filter_y) {
							for (int filter_x = 0; filter_x < l.size; ++filter_x) {
								const int in_x = in_x_origin + filter_x;
								const int in_y = in_y_origin +  filter_y;
								// If the location is outside the bounds of the input image,
								// use zero as a default value.
								if ((in_x >= 0) && (in_x < l.w) && (in_y >= 0) && (in_y < l.h)) {
									int input_val = input_data[offset(l.batch, l.c, l.h, l.w, batch, in_channel, in_y, in_x)];
									int filter_val =
											filter_data[offset(l.n, l.c, l.size, l.size, out_channel, in_channel,
															  filter_y, filter_x)];
									acc += (filter_val + filter_offset) * (input_val + input_offset);
								}
							}
						}
					}
					if (bias_data) {
						acc += bias_data[out_channel];
					}
					acc = acc * conv_output_multiplier;
					acc += conv_output_offset;
//					if (acc < 0.0f || acc > 255.0f)
//						fprintf(stderr, "[out range(0, 255)]: %.6f\n", acc);
//					acc = MAX(acc, output_activation_min);
//					acc = MIN(acc, output_activation_max);
					if (acc < conv_output_offset) {
						acc = activation_output_multiplier * (acc - conv_output_offset) *
								(quant_leaky_factor - leaky_factor_offset) + activation_output_offset;
					}
					output_data[offset(l.batch, l.out_c, l.out_h, l.out_w, batch, out_channel, out_y, out_x)] = cast_to_uint8(round(acc));
				}
			}
		}
	}
	dequantize_uint8(output_data, l.batch*l.out_c*l.out_h*l.out_w, activation_output_scale, activation_output_offset, l.output);

	if (quant_dbg) {
		float *temp_float_output = calloc(l.batch*l.outputs, sizeof(float));

		int i, j;
		fill_cpu(l.outputs*l.batch, 0, temp_float_output, 1);
		int m = l.n/l.groups;
		int k = l.size*l.size*l.c/l.groups;
		int n = l.out_w*l.out_h;
		for(i = 0; i < l.batch; ++i){
			for(j = 0; j < l.groups; ++j){
				float *a = l.weights + j*l.nweights/l.groups;
				float *b = net.workspace;
				float *c = temp_float_output + (i*l.groups + j)*n*m;
				float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

				if (l.size == 1) {
					b = im;
				} else {
					im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
				}
				gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
			}
		}

		add_bias(temp_float_output, l.biases, l.batch, l.n, l.out_h*l.out_w);
		activate_array(temp_float_output, l.outputs*l.batch, l.activation);

		int cnt = 0;
		float error = 0.0f, max_error = FLT_MIN, min_error = FLT_MAX, total_error = 0.0f;
		int total = l.batch*l.out_c*l.out_h*l.out_w;
		int cnt_error = 0;
		float fmin = FLT_MAX, fmax = FLT_MIN;
		for (int i=0; i<total; i++) {
			if (i != 0.0f) {
				error = fabs(l.output[i] - temp_float_output[i]);
				min_error = MIN(error, min_error);
				max_error = MAX(error, max_error);
				total_error += error;
				if (error > 3.f) {
					++cnt_error;
					if (cnt_error < 20) {
						fprintf(stderr, "float: %.6f, qu/uint8/de: %.6f\n", temp_float_output[i], l.output[i]);
					}
				}
				fmin = MIN(fmin, l.output[i]);
				fmax = MAX(fmax, l.output[i]);
			}
		}
		fprintf(stderr, "[quant_conv error] max: %.6f, fmin: %.6f, fmax: %.6f, err_cnt: %d, total: %d\n",
				max_error, fmin, fmax, cnt_error, total);
		free(temp_float_output);
	}
 //   activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_quant_convolutional_layer(quant_convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates + j*l.nweights/l.groups;

            float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_quant_convolutional_layer(quant_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}

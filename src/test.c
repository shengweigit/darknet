#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "quant.h"

void test_quant()
{
	int batch=2, h=4, w=4, c=2, n=3, groups=1, size=3, stride=1, padding=0;
	ACTIVATION act=RELU;
	quant_convolutional_layer quant_conv;
	quant_conv = make_quant_convolutional_layer(batch, h, w, c, n, groups, size, stride, padding, act, 0, 0, 0, 0);

	network net;
}


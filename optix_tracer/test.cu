

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void dddd(
	float3* test,
	int h, 
	int w
)
{	
	printf("----------------------------------------------------------------------\n");
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			float3 f = test[i * w + j];
			printf("%f |", f.x);
		}
		printf("\n");
	}
	printf("----------------------------------------------------------------------\n");
}

void preprocess(
	float3* test,
	int h, 
	int w
){
	dddd <<<1, 1>>>(
		test,
		h, w
	);
}

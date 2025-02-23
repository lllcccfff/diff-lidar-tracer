

#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

void preprocess(
	float3* test, int h, int w
);

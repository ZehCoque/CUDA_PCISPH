#pragma once
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ int getGlobalIdx_3D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
};

__device__ int getGlobalIdx_1D_1D() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_3D_1D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

__device__ float Poly6_Kernel(float r, float h, float pi)
{
	return 315 / (64 * pi * powf(pi, 9)) * powf(powf(r, 2) - powf(r, 2), 3);
}